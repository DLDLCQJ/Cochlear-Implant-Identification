import torch
import torch.nn as nn
import timm
import torchvision.models as tvmodels

class Loading_pretrained(nn.Module):
    """
     ImageNet, at a resolution of 128x128 or 224x224 pixels
    """
    def __init__(self, network,  num_classes, hidden_size,input_size=[64,3,128,128],pretrained=True):
        super(Loading_pretrained, self).__init__()
        self.network = network
        self.input_size = input_size
    #def configure_model(self, num_classes, input_size):
        if network == 'DeiT_B16_Distilled':
            self.model = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
        elif network == 'CLIP':
            self.model = timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_320', pretrained=pretrained,
                                           num_classes = 0)
        elif network == 'VGG':
            self.model = timm.create_model('vgg19_bn', pretrained=pretrained)
        elif network == 'Resnet':
            self.model = timm.create_model('resnet50d', pretrained=pretrained)
        elif network == 'Densenet':
            self.model = timm.create_model('densenet169', pretrained=pretrained)
        elif network == 'Inception':
            self.model = timm.create_model('inception_v3', pretrained=pretrained)
        elif network == 'Alexnet':
            self.model = tvmodels.alexnet(pretrained=pretrained, progress = False)
        elif network == 'Googlenet':
            self.model = tvmodels.googlenet(pretrained= pretrained, progress= False)
        elif network == 'Mobilenet':
            self.model = tvmodels.mobilenet_v2(pretrained=pretrained, progress = False)
        #
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # Freeze pretrained layers
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        # Modify classification layer/head(Adapting)
        if network == 'ViT':
            self.model.head = linear_block(self.model.head.in_features,num_classes)
            self.model.head_dist = linear_block(self.model.head_dist.in_features, num_classes)
        # elif network == 'CLIP':
        elif network == 'VGG': #The classification part consists of multiple fully connected layers
            in_features = self.model.get_classifier().in_features
            self.model.reset_classifier(num_classes=num_classes, global_pool='avg')
        elif network in ['Alexnet','Mobilenet']: #The classification part consists of multiple fully connected layers
            self.model_features = self.model.features
            in_features = self.avgpool(self.model_features(torch.rand(*self.input_size))).shape[1]
            self.fc = linear_block(in_features, num_classes,hidden_size,False,nn.SELU)
            self.model.classifier = self.fc
        elif network == 'Densenet': #The classification part consists of multiple fully connected layers
            in_features = self.model.classifier.in_features
            self.model.classifier = linear_block(in_features, num_classes,hidden_size,False,nn.SELU)
        elif network == 'Googlenet':
            in_features = self.model.fc.in_features
            self.fc = linear_block(in_features, num_classes,hidden_size,False,nn.SELU)
            self.model.fc = self.fc
        else:
            in_features = self.model.get_classifier().in_features
            self.fc = linear_block(in_features, num_classes,hidden_size,False,nn.SELU)
            self.model.fc = self.fc
            #self.alpha = nn.Parameter(torch.tensor(0.5)) #self.model.fc[-1].weight.squeeze()
            # self.alpha = nn.Parameter(torch.rand(self.model.fc[0].weight.shape[0],self.model.fc[0].weight.shape[0])) #self.model.fc[-1].weight.squeeze()
            # self.model.alpha = self.alpha

    # Modify the classification head
    def forward(self, x):
        if self.network == 'ViT':
            # Pass the input through the backbone to get both CLS and distillation tokens
            features = self.model.forward_features(x)
            cls_token = features[:, 0]        # First token is the CLS token
            dist_token = features[:, 1]       # Second token is the distillation token
            # Pass the tokens through their respective heads
            cls_output = self.model.head(cls_token)
            dist_output = self.model.head_dist(dist_token)
            # Combine the outputs (you could also return them separately, if needed)
            output = (cls_output + dist_output) / 2

        elif self.network == 'CLIP':
            output = self.model.forward_features(x)
            output = self.model.forward_head(output, pre_logits=True)

        elif self.network in ['Alexnet','Mobilenet']:
            output = self.model_features(x)
            #print(output.shape)
            output = self.model.classifier(self.avgpool(output).squeeze())

        else:
            # # Print a summary using torchinfo (uncomment for actual output)
            # summary(model=self.model,
            #         input_size=(64, 3, 128, 128), # make sure this is "input_size", not "input_shape"
            #         # col_names=["input_size"], # uncomment for smaller output
            #         col_names=["input_size", "output_size", "num_params", "trainable"],
            #         col_width=20,
            #         row_settings=["var_names"]
            # )
            output = self.model(x)

        return output
