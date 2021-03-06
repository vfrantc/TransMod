import torch
import torch.nn.functional as F

# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model     # extract the whole model
        self.layer_name_mapping = {     # layers that we use for the perceptual loss
            '3': "relu1_2", # layer 3
            '8': "relu2_2", # layer 8
            '15': "relu3_3" # layer 15
        }

    def output_features(self, x):
        output = {} # output features
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)