# YOLOv5 YOLO-specific modules
# with binary
import argparse
import logging
import sys
from copy import deepcopy
# import scipy.io as sio
from torch import mode
from ultralytics.utils.tal import make_anchors, dist2bbox

from ultralytics.nn.modules import DFL

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

# yolov8的模块
# from ultralytics.nn.modules.head import *
# from ultralytics.nn.modules.conv import *
# from ultralytics.nn.modules.utils import *
# from ultralytics.nn.modules.block import *
# from ultralytics.nn.modules.transformer import *

from GHOST_models.common import *
# from models.swin_transformer import *
from GHOST_models.experimental import *

# from models.edsr import EDSR
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
import scipy.io as sio
import numpy
# from models import build_model
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
import torch.nn.functional as F


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=20, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * 1, 1) for x in ch)  # .cuda()  # output conv

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class Model(nn.Module):
    export = False  # onnx export
    def __init__(self, cfg='yolov8s.yaml',input_mode='RGB',ch_steam=3, ch=3, nc=None, anchors=None,config=None,sr=False,factor=1):  #att=False,sr_att=False model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='cp936', errors='ignore') as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        self.sr = sr
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # self.input_mode = input_mode
        if input_mode == 'RGB+IR+fusion':
            self.steam, _ = parse_model(deepcopy(self.yaml),'steam', ch=[ch_steam],config=config)  # zjq model, savelist
        self.model, self.save = parse_model(deepcopy(self.yaml),'backbone+head', ch=[ch],config=config)  # model, savelist
        if self.sr == True:
            # from models.deeplab import DeepLab
            from GHOST_models.deeplabedsr import DeepLab
            if input_mode == 'IR' or input_mode == 'RGB':
                self.model_up = DeepLab(3,self.yaml['c1'],self.yaml['c2'],factor=factor)#.cuda() #'if the size is m:192,768 l:256,1024 x:320 1280
            else:
                self.model_up = DeepLab(4,self.yaml['c1'],self.yaml['c2'],factor=factor)#.cuda() #'if the size is m:192,768 l:256,1024 x:320 1280
            self.l1=self.yaml['l1']
            self.l2=self.yaml['l2']
        
        # self.f1=self.yaml['f1']  #蒸馏特征层层数
        # self.f2=self.yaml['f2']
        # self.f3=self.yaml['f3']


        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            #m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_steam, s, s),torch.zeros(1, ch_steam, s, s),input_mode)[0]])  # forward
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_steam, s, s),torch.zeros(1, ch_steam, s, s),input_mode)[0]])  # forward
            # m.anchors /= m.stride.view(-1, 1, 1)
            # check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
        # m = self.model[-2]  # Detect()
        # if isinstance(m, Detect):
        #     s = 256  # 2x min stride
        #     #m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_steam, s, s),torch.zeros(1, ch_steam, s, s),input_mode)[0]])  # forward
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_steam, s, s),torch.zeros(1, ch_steam, s, s),input_mode)[0]])  # forward
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     check_anchor_order(m)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')
        
    
    def forward(self, x, ir=torch.randn(1,3,512,512), input_mode='RGB+IR', augment=False, profile=False):
        # input_mode = 'RGB+IR' #IRRGB
        if input_mode=='RGB':
            ir=x
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                iri = scale_img(ir.flip(fi) if fi else ir, si, gs=int(self.stride.max()))
                if input_mode =='RGB+IR+fusion':
                    steam1 = self.forward_once(x,'steam',profile)
                    steam2 = self.forward_once(ir,'steam',profile)
                    steam = torch.cat([steam1,steam2],1)
                if input_mode == 'RGB+IR':
                    steam = torch.cat([xi,iri[:,0:1,:,:]],1)
                if input_mode == 'RGB':
                    steam = xi
                if input_mode == 'IR':
                    steam = iri#steam = iri[:,0:1,:,:]
                if input_mode == 'RGB+IR+SAM' or input_mode == 'RGB+IR+SSWT' or input_mode == 'RGB+IR+SSTN':
                    steam = [x,ir] #[:,0:1,:,:]
                yi = self.forward_once(steam,'yolo')[0]  # forward
                # yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            if input_mode =='RGB+IR+fusion':
                steam1 = self.forward_once(x,'steam',profile)
                steam2 = self.forward_once(ir,'steam',profile)
                steam = torch.cat([steam1,steam2],1)
                # sio.savemat('features/output.mat', mdict={'data':steam.cpu().numpy()})
            if input_mode == 'RGB+IR':
                steam = torch.cat([x,ir[:,0:1,:,:]],1)
            if input_mode == 'RGB':
                steam = x
            if input_mode == 'IR':
                steam = ir#steam = ir[:,0:1,:,:]
            if input_mode == 'RGB+IR+SAM' or input_mode == 'RGB+IR+SSWT' or input_mode == 'RGB+IR+SSTN':
                steam = [x,ir] #[:,0:1,:,:]
                
            
            self.training |= self.export

            if self.training==True:
                if self.sr:
                    y,output_sr,features = self.forward_once(steam,'yolo', profile) #zjq
                    return y,output_sr,features
                else:
                    y,features = self.forward_once(steam,'yolo', profile) #zjq
                    return y,features
            else:
                y,features = self.forward_once(steam,'yolo', profile) #zjq
                return y[0],y[1],features




    
    def forward_once(self, x, string, profile=False):
        y, dt = [], []  # outputs
        if string == 'steam':
            for m in self.steam:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                if profile:
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    t = time_synchronized()
                    for _ in range(10):
                        _ = m(x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run
                #y.append(x if m.i in self.save_steam else None)  # save output
            return x
        elif string == 'yolo': 
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
                if profile:
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    t = time_synchronized()
                    for _ in range(10):
                        _ = m(x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
                x = m(x)  # run
                y.append(x)
                

            # for feature in y[:-1]:
            #     print((torch.numel(feature)-torch.count_nonzero(feature))/torch.numel(feature))


            self.training |= self.export
            if self.training==True:
                if self.sr:
                    output_sr = self.model_up(y[self.l1],y[self.l2]) #在超分上加attention    
                    return x,output_sr,y#(y[self.f1],y[self.f2],y[self.f3])#(y[4],y[8],y[18],y[21],y[24])#(y[7],y[15],y[-2])
                else:
                    return x,y#(y[self.f1],y[self.f2],y[self.f3])#(y[4],y[8],y[18],y[21],y[24])#(y[7],y[15],y[-2])
            else:
                return x,y#(y[17],y[20],y[23])#(y[4],y[8],y[18],y[21],y[24])#(y[7],y[15],y[-2])(y[-4],y[-3],y[-2])


    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(1, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if (type(m) is Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

def parse_model(d, string, ch,config):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    nc = d['nc']
    no = 1 * (nc + 5)  # number of outputs = anchors * (classes + 5) v8中每一个grid cell只有一个anchor

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    i_shoud_add = 0

    stri = string.split('+')
    if len(stri)==2:
        string_1 = stri[0] # backbone
        string_2 = stri[1] # head
        d_ = d[string_1] + d[string_2]
        save.extend([2,4,5,6,8,9]) #save some layer of backbone
    else:
        d_ = d[stri[-1]]
    
    for i, (f, n, m, args) in enumerate(d_):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        # n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, ACmix, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, BottleneckCSP2, SPPCSP, C3, AttentionModel, C2f, SPPF]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            # c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3,BottleneckCSP2, SPPCSP, C2f]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:# or m is SAM:
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f if f < 0 else f + 1] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f if f < 0 else f + 1] // args[0] ** 2
        else:
            c2 = ch[f if f < 0 else f + 1]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i+i_shoud_add, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i+i_shoud_add, f, n, np, t, args))  # print
        save.extend(x % (i+i_shoud_add) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)

        
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
