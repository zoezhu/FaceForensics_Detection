import io, random
import logging
import json
from collections import OrderedDict
from PIL import Image

import numpy as np
import torch
from torch import nn as nn


def write_json(data, file_path):
    with open(file_path, 'w') as out:
        json.dump(data, out, indent=4)


def copy_file(src_path, dest_path):
    with open(src_path) as src:
        with open(dest_path, 'w') as dest:
            dest.write(src.read())


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    def get_shape(output):
        if isinstance(output, (list, tuple)):
            shape = [get_shape(o) for o in output]
        else:
            shape = list(output.size())

            for i in range(len(shape)):
                if shape[i] == batch_size:
                    shape[i] = -1
                    return shape
        return shape

    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            if class_name in ('InceptionResnetV1', 'CNN_LSTM'):
                return

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = get_shape(output)
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = sum([m.numel() for m in module.parameters()])
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [ torch.rand(2, *in_size).type(dtype).to(device=device) for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    trainable_params, total_params  = count_parameters(model)
    total_output = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        output_shape = summary[layer]["output_shape"]
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(output_shape),
            "{0:,}".format(summary[layer]["nb_params"]),
        )

        total_output += np.prod(output_shape[0] if isinstance(output_shape[0], list) else output_shape)
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
    return total_params, trainable_params


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
    print(f"Freeze the net until {param_name}")


def adjust_lr(optimizer, decay_rate=0.9):
    for g in optimizer.param_groups:
        g['lr'] *= decay_rate
        new_lr = g['lr']
    return new_lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_loger(log_path):
    logger = logging.getLogger('myLogger')
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def model_validation(net, dataloader, criterion):
    net.eval()
    all_predictions= []
    real_predictions = []
    fake_predictions = []
    all_targets = []
    loss_values = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.cuda()
            label = label.cuda()
            output = net(data).squeeze()
            bz = data.shape[0]
            # Loss
            loss = criterion(output, label)
            loss_values.append(loss.item())

            predictions = output > 0.0
            all_predictions.append(predictions)
            all_targets.append(label)

            real_id = label == 1
            real_predictions.append(predictions[real_id])
            fake_id = label == 0
            fake_predictions.append(predictions[fake_id])
    
    # Acc
    all_predictions = torch.cat(all_predictions).int()
    all_targets = torch.cat(all_targets).int()
    val_acc = (all_predictions == all_targets).sum().float().item()
    val_loss = sum(loss_values) / len(loss_values)
    val_acc /= all_targets.shape[0]

    # Real Acc
    real_predictions = torch.cat(real_predictions).int()
    real_acc = (real_predictions==1).sum().float().item() / len(real_predictions)

    # Fake Acc
    fake_predictions = torch.cat(fake_predictions).int()
    fake_acc = (fake_predictions==0).sum().float().item() / len(fake_predictions)

    return val_loss, val_acc, real_acc, fake_acc


def model_validation_bce(net, dataloader, criterion):
    net.eval()
    all_predictions= []
    real_predictions = []
    fake_predictions = []
    all_targets = []
    loss_values = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.cuda()
            label = label.cuda()
            output = net(data).squeeze()
            bz = data.shape[0]
            # Loss
            loss = F.binary_cross_entropy_with_logits(output, label)
            loss_values.append(loss.item())

            predictions = output > 0.0
            all_predictions.append(predictions)
            all_targets.append(label)

            real_id = label == 1
            real_predictions.append(predictions[real_id])
            fake_id = label == 0
            fake_predictions.append(predictions[fake_id])
    
    # Acc
    all_predictions = torch.cat(all_predictions).int()
    all_targets = torch.cat(all_targets).int()
    val_acc = (all_predictions == all_targets).sum().float().item()
    val_loss = sum(loss_values) / len(loss_values)
    val_acc /= all_targets.shape[0]

    # Real Acc
    real_predictions = torch.cat(real_predictions).int()
    real_acc = (real_predictions==1).sum().float().item() / (1.*len(real_predictions))

    # Fake Acc
    fake_predictions = torch.cat(fake_predictions).int()
    fake_acc = (fake_predictions==0).sum().float().item() / (1.*len(fake_predictions))

    return val_loss, val_acc, real_acc, fake_acc


def randomJPEGcompression(image):
    qf = random.randrange(1, 25)
    outputIoStream = io.BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)
