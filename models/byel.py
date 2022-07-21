import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):

    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# regularization for ABAW
class Regressor(nn.Module):
    def __init__(self, rows, columns):
        super().__init__()
        self.rows = rows
        self.columns = columns

        self.W = nn.Parameter(torch.rand(self.rows,self.columns))
        self.batchnorm = nn.BatchNorm1d(6)
        nn.init.orthogonal_(self.W)

    def forward(self, x):
        y = torch.matmul(x,self.W)
        y = self.batchnorm(y)
        return y, self.W


# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)
        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        # print(f'projection : {projection.shape}')
        return projection, representation

# main class

class BootstropOnEmotionLatentRepresentation(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        class_num = 6
    ):
        super().__init__()
        self.net = net
        self.projection_size = projection_size
        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        self.regressor = Regressor(projection_size, class_num)

        self.cross_entrophy = nn.CrossEntropyLoss()

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, label = None, representation = None, return_embedding = False, return_projection = True):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)
        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, online_representation_one = self.online_encoder(image_one)
        online_proj_two, online_representation_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)


        label_one,_ = self.regressor(online_pred_one)
        label_two,w = self.regressor(online_pred_two)

        emotion_vector = 0
        if(type(label) != type(None)):

            emotion_vector = w[:,label].permute(1,0)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        # regressor orthonomal weights loss
        wtw = torch.matmul(w.permute(1, 0), w)

        differ_orthonomal = torch.eye(w.shape[-1]).cuda() - wtw
        loss_w = torch.abs(differ_orthonomal)

        # emotion classification loss
        classification_loss = torch.Tensor([0]).cuda()
        representation_loss = torch.Tensor([0]).cuda()
        if(type(label) != type(None)):
            classification_loss_one = self.cross_entrophy(label_one,label)
            classification_loss_two = self.cross_entrophy(label_two,label)
            classification_loss = classification_loss_one + classification_loss_two

            # representation = self.online_predictor(representation)
            representation_loss_one = loss_fn(online_representation_one, representation.detach())
            representation_loss_two = loss_fn(online_representation_two, representation.detach())

            representation_loss = (representation_loss_one + representation_loss_two).mean()


        # byol loss
        scale_one = torch.norm(online_pred_one,p=2,dim=-1)
        scale_two = torch.norm(online_pred_two,p=2,dim=-1)
        #
        # print(scale_one,scale_two)

        online_pred_one = online_pred_one - scale_one.unsqueeze(-1) * emotion_vector
        online_pred_two = online_pred_two - scale_two.unsqueeze(-1) * emotion_vector


        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        byol_loss = loss_one + loss_two

        return byol_loss.mean(), loss_w.sum(), classification_loss, representation_loss


class BootstropOnEmotionLatent(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        class_num = 6
    ):
        super().__init__()
        self.net = net
        self.projection_size = projection_size
        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        self.regressor = Regressor(projection_size, class_num)

        self.cross_entrophy = nn.CrossEntropyLoss()

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, label = None, return_embedding = False, return_projection = True):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)
        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, online_representation_one = self.online_encoder(image_one)
        online_proj_two, online_representation_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)


        label_one,_ = self.regressor(online_pred_one)
        label_two,w = self.regressor(online_pred_two)

        emotion_vector = 0
        if(type(label) != type(None)):

            emotion_vector = w[:,label].permute(1,0)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        # regressor orthonomal weights loss
        wtw = torch.matmul(w.permute(1, 0), w)

        differ_orthonomal = torch.eye(w.shape[-1]).cuda() - wtw
        loss_w = torch.abs(differ_orthonomal)

        # emotion classification loss
        classification_loss = torch.Tensor([0]).cuda()
        if(type(label) != type(None)):
            classification_loss_one = self.cross_entrophy(label_one,label)
            classification_loss_two = self.cross_entrophy(label_two,label)
            classification_loss = classification_loss_one + classification_loss_two

        # byol loss
        scale_one = torch.norm(online_pred_one,p=2,dim=-1)
        scale_two = torch.norm(online_pred_two,p=2,dim=-1)
        #
        # print(scale_one,scale_two)

        online_pred_one = online_pred_one - scale_one.unsqueeze(-1) * emotion_vector
        online_pred_two = online_pred_two - scale_two.unsqueeze(-1) * emotion_vector


        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        byol_loss = loss_one + loss_two

        return byol_loss.mean(), loss_w.sum(), classification_loss


if __name__ == '__main__':

    print(f'loss : {nn.CrossEntropyLoss()(torch.Tensor([[0,0,100.0],[0,100.0,0]]),torch.Tensor([2,1]).long())}')

    from models.ResNet import resnet50

    resnet = resnet50().cuda()

    learner = BootstropOnEmotionLatent(
        resnet,
        image_size=128,
        hidden_layer='avgpool'
    ).cuda()

    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)


    def sample_unlabelled_images():
        return torch.randn(20, 3, 128, 128).cuda(), torch.randint(5,[20]).cuda()
    def sample_unlabelled_images2():
        return torch.randn(20, 3, 112, 112).cuda(), torch.randint(5,[20]).cuda()
    scaler = GradScaler()

    for _ in range(100):
        images, label = sample_unlabelled_images()
        print(images.shape,label.shape)
        with autocast():
            byol_loss,loss_w,classification_loss = learner(images,label)
            loss = byol_loss + loss_w + classification_loss

        optimizer.zero_grad()

        print(byol_loss.shape,loss_w.shape,classification_loss.shape)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f'{_} EPOCH')
        learner.update_moving_average()  # update moving average of target encoder

    # save your improved network
    torch.save(resnet.state_dict(), './improved-net.pt')

    # x = torch.randn(3,2)
    # print(x)
    # print(F.normalize(x,2,dim=1))
    # print(F.normalize(x, 2, dim=0))
    # print(x[-1])
    # print(x[-1]/torch.sqrt((x[-1][0])**2 + (x[-1][1])**2))

