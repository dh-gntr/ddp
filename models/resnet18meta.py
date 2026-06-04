import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random
'''
resnet18 base
'''

class ResNet18_AllFeatures(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18_AllFeatures, self).__init__()
        base = models.resnet18(pretrained=pretrained)

        # keep original layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        features = []  # collect features after each conv/fc

        # ---- stem ----
        x = self.conv1(x)#; features.append(x)   # layer 1
        x = self.bn1(x)#; features.append(x)     # layer 2
        x = self.relu(x)#; features.append(x)    # layer 3
        x = self.maxpool(x)                     # no params (not counted)

        # ---- layer1 (2 BasicBlocks × 2 convs = 4 convs) ----
        for block in self.layer1:
            x = block.conv1(x); features.append(x)  # layers 4,6,...
            x = block.bn1(x)
            x = block.relu(x)
            x = block.conv2(x); features.append(x)
            x = block.bn2(x)
            if block.downsample is not None:
                identity = block.downsample(x)
            x = x + identity if block.downsample is not None else x
            x = block.relu(x)

        # ---- layer2 ----
        for block in self.layer2:
            identity = x
            out = block.conv1(x); features.append(out)
            out = block.bn1(out)
            out = block.relu(out)
            out = block.conv2(out); features.append(out)
            out = block.bn2(out)
            if block.downsample is not None:
                identity = block.downsample(x)
            x = out + identity
            x = block.relu(x)

        # ---- layer3 ----
        for block in self.layer3:
            identity = x
            out = block.conv1(x); features.append(out)
            out = block.bn1(out)
            out = block.relu(out)
            out = block.conv2(out); features.append(out)
            out = block.bn2(out)
            if block.downsample is not None:
                identity = block.downsample(x)
            x = out + identity
            x = block.relu(x)

        # ---- layer4 ----
        for block in self.layer4:
            identity = x
            out = block.conv1(x); features.append(out)
            out = block.bn1(out)
            out = block.relu(out)
            out = block.conv2(out); features.append(out)
            out = block.bn2(out)
            if block.downsample is not None:
                identity = block.downsample(x)
            x = out + identity
            x = block.relu(x)

        # ---- classifier ----
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #features.append(x)   # layer 17: pooled feature vector
        out = self.fc(x)
        #features.append(out) # layer 18: final linear output

        return out, features

class ResNet18_BaseModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18_BaseModel, self).__init__()
        base_model = models.resnet18(pretrained=pretrained)
        
        # Extract layers
        self.layer1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool, base_model.layer1)
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        x = self.avgpool(f4)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        # Return prediction and 4 intermediate features
        return out, (f1, f2, f3, f4)


'''
Bach Dataset Resnet 18 Meta Model
'''

class Resnet18_meta_model(nn.Module):
    # def __init__(self, fea_dim1, fea_dim2, fea_dim3, fea_dim4):
    #     super(Resnet18_meta_model, self).__init__()
    #     self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
    #     self.classifier1_fc1 = nn.Linear(fea_dim1, 512)
    #     self.classifier1_fc2 = nn.Linear(256, 4)

    #     self.classifier2_fc1 = nn.Linear(fea_dim2, 512)
    #     self.classifier2_fc2 = nn.Linear(256, 4)

    #     self.classifier3_fc1 = nn.Linear(fea_dim3, 512)
    #     self.classifier3_fc2 = nn.Linear(256, 4)

    #     self.classifier4_fc1 = nn.Linear(fea_dim4, 512)
    #     self.classifier4_fc2 = nn.Linear(256, 4)


    #     self.classifier_final = nn.Linear(4 * 4, 4)

    # def forward(self, fea1, fea2, fea3, fea4):
    #     fea1 = F.relu(self.classifier1_fc1(fea1))
    #     fea1 = self.pooling(fea1)
    #     fea1 = F.relu(self.classifier1_fc2(fea1))

    #     fea2 = F.relu(self.classifier2_fc1(fea2))
    #     fea2 = self.pooling(fea2)
    #     fea2 = F.relu(self.classifier2_fc2(fea2))

    #     fea3 = F.relu(self.classifier3_fc1(fea3))
    #     fea3 = self.pooling(fea3)
    #     fea3 = F.relu(self.classifier3_fc2(fea3))

    #     fea4 = F.relu(self.classifier4_fc1(fea4))
    #     fea4 = self.pooling(fea4)
    #     fea4 = F.relu(self.classifier4_fc2(fea4))

    #     fea = torch.cat((fea1, fea2, fea3, fea4,), 1)
    #     z = self.classifier_final(fea)
    #     return z
    def __init__(self, fea_dim1, fea_dim2, fea_dim3, fea_dim4):#,fea_dim5):   #uncommented   14th oct 24 gora
#     def __init__(self, fea_dim2, fea_dim3, fea_dim4):            #commented
        super(Resnet18_meta_model, self).__init__()               
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.classifier1_fc1 = nn.Linear(fea_dim1, 1024)          #uncommented    14th oct g 
        # self.classifier1_fc2 = nn.Linear(512, 10)  
        
        # self.classifier1b_fc1 = nn.Linear(fea_dim1, 1024)          #uncommented    14th oct g 
        # self.classifier1b_fc2 = nn.Linear(512, 10)   # uncommented   14th oct g # 4 --> 10

        # self.classifier1c_fc1 = nn.Linear(fea_dim1, 1024)          #uncommented    14th oct g 
        # self.classifier1c_fc2 = nn.Linear(512, 10)

        # self.classifier1d_fc1 = nn.Linear(fea_dim1, 1024)          #uncommented    14th oct g 
        # self.classifier1d_fc2 = nn.Linear(512, 10)

        self.classifier2_fc1 = nn.Linear(fea_dim2, 2048)
        self.classifier2_fc2 = nn.Linear(1024, 512)
        self.classifier2_fc3 = nn.Linear(256, 10)  # 4 --> 10

        # self.classifier2b_fc1 = nn.Linear(fea_dim2, 2048)
        # self.classifier2b_fc2 = nn.Linear(1024, 512)
        # self.classifier2b_fc3 = nn.Linear(256, 10)  # 4 --> 10

        self.classifier3_fc1 = nn.Linear(fea_dim3, 1024)
        self.classifier3_fc2 = nn.Linear(512, 10)  # 4 --> 10

        # self.classifier3b_fc1 = nn.Linear(fea_dim3, 1024)
        # self.classifier3b_fc2 = nn.Linear(512, 10)  # 4 --> 10

        # self.classifier4_fc1 = nn.Linear(fea_dim4, 2048)
        # self.classifier4_fc2 = nn.Linear(1024, 512)
        # self.classifier4_fc3 = nn.Linear(256, 10)  # 4 --> 10

        # self.classifier4b_fc1 = nn.Linear(fea_dim4, 2048)
        # self.classifier4b_fc2 = nn.Linear(1024, 512)
        # self.classifier4b_fc3 = nn.Linear(256, 10)  # 4 --> 10
        
#         self.classifier5_fc1 = nn.Linear(fea_dim5, 1024)
#         self.classifier5_fc2 = nn.Linear(512, 10)  # 4 --> 10



        self.classifier_final = nn.Linear(1 * 10, 10)              # 3-->4 on 14th Oct 24 by gora # 4 --> 10

    def forward(self, fea1, fea2, fea3, fea4, idx):#, fea5):              #fea1 has been added to the arguments
        #print(fea4.shape)
        # proj_id = random.choice([1,2,3,4])
        # if proj_id == 1:
        # fea1 = fea1.reshape(fea1.size(0),-1)
        # fea1 = F.relu(self.classifier1_fc1(fea1))           #uncommented on 14th Oct 24 by gora
        # fea1 = self.pooling(fea1)                           #       "
        # fea1 = F.relu(self.classifier1_fc2(fea1))           #       " 
        fea1 = fea1.reshape(fea1.size(0), -1)
        fea1 = F.relu(self.classifier2_fc1(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier2_fc2(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier2_fc3(fea1))

        # elif proj_id == 2:
        #     fea1 = fea1.reshape(fea1.size(0),-1)
        #     fea1 = F.relu(self.classifier1c_fc1(fea1))           #uncommented on 14th Oct 24 by gora
        #     fea1 = self.pooling(fea1)                           #       "
        #     fea1 = F.relu(self.classifier1c_fc2(fea1))  

        # elif proj_id == 3:
        #     fea1 = fea1.reshape(fea1.size(0),-1)
        #     fea1 = F.relu(self.classifier1d_fc1(fea1))           #uncommented on 14th Oct 24 by gora
        #     fea1 = self.pooling(fea1)                           #       "
        #     fea1 = F.relu(self.classifier1d_fc2(fea1))  
            

        # else:
        #     fea1 = fea1.reshape(fea1.size(0),-1)
        #     fea1 = F.relu(self.classifier1b_fc1(fea1))           #uncommented on 14th Oct 24 by gora
        #     fea1 = self.pooling(fea1)                           #       "
        #     fea1 = F.relu(self.classifier1b_fc2(fea1))   
            # fea1 = fea1.reshape(fea1.size(0), -1)
            # fea1 = F.relu(self.classifier2b_fc1(fea1))
            # fea1 = self.pooling(fea1)
            # fea1 = F.relu(self.classifier2b_fc2(fea1))
            # fea1 = self.pooling(fea1)
            # fea1 = F.relu(self.classifier2b_fc3(fea1))

        # proj_id = random.choice([1,2,3,4])
        # if proj_id == 1:
        # fea2 = fea2.reshape(fea2.size(0),-1)
        # fea2 = F.relu(self.classifier1_fc1(fea2))           
        # fea2 = self.pooling(fea2)                          
        # fea2 = F.relu(self.classifier1_fc2(fea2)) 
            # fea2 = fea2.reshape(fea2.size(0), -1)
            # fea2 = F.relu(self.classifier2b_fc1(fea2))
            # fea2 = self.pooling(fea2)
            # fea2 = F.relu(self.classifier2b_fc2(fea2))
            # fea2 = self.pooling(fea2)
            # fea2 = F.relu(self.classifier2b_fc3(fea2))

        # elif proj_id == 2:
        #     fea2 = fea2.reshape(fea2.size(0),-1)
        #     fea2 = F.relu(self.classifier1c_fc1(fea2))           
        #     fea2 = self.pooling(fea2)                          
        #     fea2 = F.relu(self.classifier1c_fc2(fea2))

        # elif proj_id == 3:
        #     fea2 = fea2.reshape(fea2.size(0),-1)
        #     fea2 = F.relu(self.classifier1d_fc1(fea2))           
        #     fea2 = self.pooling(fea2)                          
        #     fea2 = F.relu(self.classifier1d_fc2(fea2)) 
            
        # else:  
        #     fea2 = fea2.reshape(fea2.size(0),-1)
        #     fea2 = F.relu(self.classifier1b_fc1(fea2))           
        #     fea2 = self.pooling(fea2)                          
        #     fea2 = F.relu(self.classifier1b_fc2(fea2))  
        fea2 = fea2.reshape(fea2.size(0), -1)
        fea2 = F.relu(self.classifier2_fc1(fea2))
        fea2 = self.pooling(fea2)
        fea2 = F.relu(self.classifier2_fc2(fea2))
        fea2 = self.pooling(fea2)
        fea2 = F.relu(self.classifier2_fc3(fea2))

        # proj_id = random.choice([1,2,3,4])
        # if proj_id == 1:
            
        #     fea3 = fea3.reshape(fea3.size(0), -1)
        #     fea3 = F.relu(self.classifier1_fc1(fea3))
        #     fea3 = self.pooling(fea3)
        #     fea3 = F.relu(self.classifier1_fc2(fea3))
            # fea3 = fea3.reshape(fea3.size(0), -1)
            # fea3 = F.relu(self.classifier2_fc1(fea3))
            # fea3 = self.pooling(fea3)
            # fea3 = F.relu(self.classifier2_fc2(fea3))
            # fea3 = self.pooling(fea3)
            # fea3 = F.relu(self.classifier2_fc3(fea3))
        fea3 = fea3.reshape(fea3.size(0),-1)
        fea3 = F.relu(self.classifier3_fc1(fea3))
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc2(fea3))
        # elif proj_id == 2:
            # fea3 = fea3.reshape(fea3.size(0), -1)
            # fea3 = F.relu(self.classifier1c_fc1(fea3))
            # fea3 = self.pooling(fea3)
            # fea3 = F.relu(self.classifier1c_fc2(fea3))

        # elif proj_id == 3:
        #     fea3 = fea3.reshape(fea3.size(0), -1)
        #     fea3 = F.relu(self.classifier1d_fc1(fea3))
        #     fea3 = self.pooling(fea3)
        #     fea3 = F.relu(self.classifier1d_fc2(fea3))
            
        # else:
        #     fea3 = fea3.reshape(fea3.size(0), -1)
        #     fea3 = F.relu(self.classifier1b_fc1(fea3))
        #     fea3 = self.pooling(fea3)
        #     fea3 = F.relu(self.classifier1b_fc2(fea3))

            # fea3 = fea3.reshape(fea3.size(0), -1)
            # fea3 = F.relu(self.classifier2b_fc1(fea3))
            # fea3 = self.pooling(fea3)
            # fea3 = F.relu(self.classifier2b_fc2(fea3))
            # fea3 = self.pooling(fea3)
            # fea3 = F.relu(self.classifier2b_fc3(fea3))
            # fea3 = fea3.reshape(fea3.size(0),-1)
            # fea3 = F.relu(self.classifier3b_fc1(fea3))
            # fea3 = self.pooling(fea3)
            # fea3 = F.relu(self.classifier3b_fc2(fea3))


        # proj_id = random.choice([1,2,3,4])
        # if proj_id == 1:
    
        #     fea4 = fea4.reshape(fea4.size(0), -1)
        #     fea4 = F.relu(self.classifier1_fc1(fea4))
        #     fea4 = self.pooling(fea4)
        #     fea4 = F.relu(self.classifier1_fc2(fea4))
            # fea4 = fea4.reshape(fea4.size(0), -1)
            # fea4 = F.relu(self.classifier2b_fc1(fea4))
            # fea4 = self.pooling(fea4)
            # fea4 = F.relu(self.classifier2b_fc2(fea4))
            # fea4 = self.pooling(fea4)
            # fea4 = F.relu(self.classifier2b_fc3(fea4))
        fea4 = fea4.reshape(fea4.size(0),-1)
        fea4 = F.relu(self.classifier3_fc1(fea4))
        fea4 = self.pooling(fea4)
        fea4 = F.relu(self.classifier3_fc2(fea4))
            # fea4 = fea4.reshape(fea4.size(0),-1)
            # fea4 = F.relu(self.classifier4b_fc1(fea4))
            # fea4 = self.pooling(fea4)
            # fea4 = F.relu(self.classifier4b_fc2(fea4))
            # fea4 = self.pooling(fea4)
            # fea4 = F.relu(self.classifier4b_fc3(fea4))

        # elif proj_id == 2:
        #     fea4 = fea4.reshape(fea4.size(0), -1)
        #     fea4 = F.relu(self.classifier1c_fc1(fea4))
        #     fea4 = self.pooling(fea4)
        #     fea4 = F.relu(self.classifier1c_fc2(fea4))

        # elif proj_id == 3:
        #     fea4 = fea4.reshape(fea4.size(0), -1)
        #     fea4 = F.relu(self.classifier1d_fc1(fea4))
        #     fea4 = self.pooling(fea4)
        #     fea4 = F.relu(self.classifier1d_fc2(fea4))
            
        # else:
            
        #     fea4 = fea4.reshape(fea4.size(0), -1)
        #     fea4 = F.relu(self.classifier1b_fc1(fea4))
        #     fea4 = self.pooling(fea4)
        #     fea4 = F.relu(self.classifier1b_fc2(fea4))
        # fea4 = fea4.reshape(fea4.size(0), -1)
        # fea4 = F.relu(self.classifier2_fc1(fea4))
        # fea4 = self.pooling(fea4)
        # fea4 = F.relu(self.classifier2_fc2(fea4))
        # fea4 = self.pooling(fea4)
        # fea4 = F.relu(self.classifier2_fc3(fea4))
            # fea4 = fea4.reshape(fea4.size(0),-1)
            # fea4 = F.relu(self.classifier3b_fc1(fea4))
            # fea4 = self.pooling(fea4)
            # fea4 = F.relu(self.classifier3b_fc2(fea4))
            # fea4 = fea4.reshape(fea4.size(0),-1)
            # fea4 = F.relu(self.classifier4_fc1(fea4))
            # fea4 = self.pooling(fea4)
            # fea4 = F.relu(self.classifier4_fc2(fea4))
            # fea4 = self.pooling(fea4)
            # fea4 = F.relu(self.classifier4_fc3(fea4))


        
        # fea1 = fea1.reshape(fea1.size(0), -1)
        # fea1 = F.relu(self.classifier2_fc1(fea1))
        # fea1 = self.pooling(fea1)
        # fea1 = F.relu(self.classifier2_fc2(fea1))
        # fea1 = self.pooling(fea1)
        # fea1 = F.relu(self.classifier2_fc3(fea1))
        
        
        
        # fea2 = fea2.reshape(fea2.size(0), -1)
        # fea2 = F.relu(self.classifier2b_fc1(fea2))
        # fea2 = self.pooling(fea2)
        # fea2 = F.relu(self.classifier2b_fc2(fea2))
        # fea2 = self.pooling(fea2)
        # fea2 = F.relu(self.classifier2b_fc3(fea2))
        

        # fea3 = fea3.reshape(fea3.size(0), -1)
        # fea3 = F.relu(self.classifier2_fc1(fea3))
        # fea3 = self.pooling(fea3)
        # fea3 = F.relu(self.classifier2_fc2(fea3))
        # fea3 = self.pooling(fea3)
        # fea3 = F.relu(self.classifier2_fc3(fea3))

        # fea4 = fea4.reshape(fea4.size(0), -1)
        # fea4 = F.relu(self.classifier2b_fc1(fea4))
        # fea4 = self.pooling(fea4)
        # fea4 = F.relu(self.classifier2b_fc2(fea4))
        # fea4 = self.pooling(fea4)
        # fea4 = F.relu(self.classifier2b_fc3(fea4))

        
        # fea3 = fea3.reshape(fea3.size(0),-1)
        # fea3 = F.relu(self.classifier3_fc1(fea3))
        # fea3 = self.pooling(fea3)
        # fea3 = F.relu(self.classifier3_fc2(fea3))
        
        
        
        # fea4 = fea4.reshape(fea4.size(0),-1)
        # fea4 = F.relu(self.classifier3b_fc1(fea4))
        # fea4 = self.pooling(fea4)
        # fea4 = F.relu(self.classifier3b_fc2(fea4))

        # fea4 = fea4.reshape(fea4.size(0),-1)
        # fea4 = F.relu(self.classifier4_fc1(fea4))
        # fea4 = self.pooling(fea4)
        # fea4 = F.relu(self.classifier4_fc2(fea4))
        # fea4 = self.pooling(fea4)
        # fea4 = F.relu(self.classifier4_fc3(fea4))

        # fea5 = fea5.reshape(fea5.size(0),-1)
        # fea5 = F.relu(self.classifier4_fc1(fea5))
        # fea5 = self.pooling(fea5)
        # fea5 = F.relu(self.classifier4_fc2(fea5))
        # fea5 = self.pooling(fea5)
        # fea5 = F.relu(self.classifier4_fc3(fea5))
        
        # fea5 = fea5.reshape(fea5.size(0),-1)
        # fea5 = F.relu(self.classifier5_fc1(fea5))
        # fea5 = self.pooling(fea5)
        # fea5 = F.relu(self.classifier5_fc2(fea5))
        
        
        f1 = None
        f2 = None
        if idx == 1:
            fea = torch.mean(torch.stack([fea1,fea2],dim=0),dim=0)
            f1 = fea1
            f2 = fea2
        if idx == 2:
            fea = torch.mean(torch.stack([fea2,fea3],dim=0),dim=0)
            f1 = fea2
            f2 = fea3
        if idx == 3:
            fea = torch.mean(torch.stack([fea3,fea4],dim=0),dim=0)
            f1 = fea3
            f2 = fea4
        if idx == 4:
            fea = torch.mean(torch.stack([fea4,fea2],dim=0),dim=0)
            f1 = fea4
            f2 = fea2
        if idx == 5:
            fea = torch.mean(torch.stack([fea1,fea3],dim=0),dim=0)
            f1 = fea1
            f2 = fea3   
        if idx == 6:
            fea = torch.mean(torch.stack([fea1,fea4],dim=0),dim=0)
            f1 = fea1
            f2 = fea4
        # if idx == 7:
        #     fea = torch.mean(torch.stack([fea1,fea5],dim=0),dim=0)
        #     f1 = fea1
        #     f2 = fea5
        # if idx == 8:
        #     fea = torch.mean(torch.stack([fea2,fea5],dim=0),dim=0)
        #     f1 = fea2
        #     f2 = fea5
        # if idx == 9:
        #     fea = torch.mean(torch.stack([fea3,fea5],dim=0),dim=0)
        #     f1 = fea3
        #     f2 = fea5
        # if idx == 10:
        #     fea = torch.mean(torch.stack([fea4,fea5],dim=0),dim=0)
        #     f1 = fea4
        #     f2 = fea5
        if idx == 7:
            fea = torch.mean(torch.stack([fea1, fea2,fea3,fea4], dim=0), dim=0)

        # if idx == 1:
        #     fea = fea1
        # if idx == 2:
        #     fea = fea2
        # if idx == 3:
        #     fea = fea3
        # if idx == 4:
        #     fea = fea4
        # if idx == 5:
        #     fea = torch.mean(torch.stack([fea1, fea2, fea3, fea4], dim=0), dim=0)
               

        fea1 = F.softmax(self.classifier_final(fea1),dim=-1)
        fea2 = F.softmax(self.classifier_final(fea2),dim=-1)
        fea3 = F.softmax(self.classifier_final(fea3),dim=-1)
        fea4 = F.softmax(self.classifier_final(fea4),dim=-1)
        #fea5 = F.softmax(self.classifier_final(fea5),dim=-1)

        feats = torch.stack([fea1,fea2,fea3,fea4],dim=0) #H,B,C
        conf,pred = torch.max(torch.mean(feats,dim=0),dim=-1)

        
        var_score = feats.var(dim=0)[torch.arange(feats.size(1)),pred]*(1-conf) #.mean(dim=1)
        #print(var_score.shape)


 #         fea = torch.cat((fea1,fea2, fea3,fea4), 1)        #uncommented on 14th Oct 24 by gora
#         fea = torch.cat((fea2, fea3, fea4,), 1)             #commented on 14th Oct 24 by gora
        z = self.classifier_final(fea)
        return F.softmax(z,dim=-1), var_score,f1,f2 #z F.softmax(z,dim=-1)
