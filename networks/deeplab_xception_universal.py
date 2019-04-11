import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn import Parameter
from networks import deeplab_xception, gcn, deeplab_xception_synBN



class deeplab_xception_transfer_basemodel_savememory(deeplab_xception.DeepLabv3_plus):
    def __init__(self, nInputChannels=3, n_classes=7, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=20, transfer_graph=None):
        super(deeplab_xception_transfer_basemodel_savememory, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                  os=os,)

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_graph' not in name \
                    and 'transpose_graph' not in name and 'middle' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def top_forward(self, input, adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)

        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        source_graph2 = self.source_graph_conv2.forward(source_graph1, adj=adj2_source, relu=True)
        source_graph3 = self.source_graph_conv2.forward(source_graph2, adj=adj2_source, relu=True)

        ### target source
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)


    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


class deeplab_xception_transfer_basemodel_savememory_synbn(deeplab_xception_synBN.DeepLabv3_plus):
    def __init__(self, nInputChannels=3, n_classes=7, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=20, transfer_graph=None):
        super(deeplab_xception_transfer_basemodel_savememory_synbn, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                  os=os,)


    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_graph' not in name \
                    and 'transpose_graph' not in name and 'middle' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def top_forward(self, input, adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)

        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        source_graph2 = self.source_graph_conv2.forward(source_graph1, adj=adj2_source, relu=True)
        source_graph3 = self.source_graph_conv2.forward(source_graph2, adj=adj2_source, relu=True)

        ### target source
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)


    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


class deeplab_xception_end2end_3d(deeplab_xception_transfer_basemodel_savememory):
    def __init__(self, nInputChannels=3, n_classes=20, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=7, middle_classes=18, transfer_graph=None):
        super(deeplab_xception_end2end_3d, self).__init__(nInputChannels=nInputChannels,
                                                          n_classes=n_classes,
                                                          os=os, )
        ### source graph
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=source_classes)
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.source_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=source_classes
                                                                   )
        self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])
        self.source_semantic = nn.Conv2d(out_channels,source_classes,1)
        self.middle_semantic = nn.Conv2d(out_channels, middle_classes, 1)

        ### target graph 1
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### middle
        self.middle_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=middle_classes)
        self.middle_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.middle_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.middle_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### multi transpose
        self.transpose_graph_source2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=n_classes)
        self.transpose_graph_target2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=source_classes)

        self.transpose_graph_middle2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)
        self.transpose_graph_middle2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)

        self.transpose_graph_source2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=middle_classes)
        self.transpose_graph_target2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=middle_classes)


        self.fc_graph_source = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_target = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_middle = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)

    def freeze_totally_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def freeze_backbone_bn(self):
        for m in self.xception_features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def top_forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer_s2t=None, adj3_transfer_t2s=None,
            adj4_middle=None,adj5_transfer_s2m=None,adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)
        ### target source
        target_graph = self.target_featuremap_2_graph(x)
        ### middle source
        middle_graph = self.middle_featuremap_2_graph(x)

        ##### end2end multi task

        ### first task
        # print(source_graph.size(),target_graph.size())
        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        target_graph1 = self.target_graph_conv1.forward(target_graph, adj=adj1_target, relu=True)
        middle_graph1 = self.target_graph_conv1.forward(middle_graph, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph1_v5 = self.transpose_graph_source2target.forward(source_graph1, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph1_v5 = self.transpose_graph_source2middle.forward(source_graph1,adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph1_v5 = self.transpose_graph_target2source.forward(target_graph1, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph1_v5 = self.transpose_graph_target2middle.forward(target_graph1, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph1_v5 = self.transpose_graph_middle2source.forward(middle_graph1, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph1_v5 = self.transpose_graph_middle2target.forward(middle_graph1, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph1 = self.similarity_trans(source_graph1, target_graph1)
        source_2_middle_graph1 = self.similarity_trans(source_graph1, middle_graph1)
        # target 2 source middle
        target_2_source_graph1 = self.similarity_trans(target_graph1, source_graph1)
        target_2_middle_graph1 = self.similarity_trans(target_graph1, middle_graph1)
        # middle 2 source target
        middle_2_source_graph1 = self.similarity_trans(middle_graph1, source_graph1)
        middle_2_target_graph1 = self.similarity_trans(middle_graph1, target_graph1)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph1 = torch.cat(
            (source_graph1, target_2_source_graph1, target_2_source_graph1_v5,
             middle_2_source_graph1, middle_2_source_graph1_v5), dim=-1)
        source_graph1 = self.fc_graph_source.forward(source_graph1, relu=True)
        # target
        target_graph1 = torch.cat(
            (target_graph1, source_2_target_graph1, source_2_target_graph1_v5,
             middle_2_target_graph1, middle_2_target_graph1_v5), dim=-1)
        target_graph1 = self.fc_graph_target.forward(target_graph1, relu=True)
        # middle
        middle_graph1 = torch.cat((middle_graph1, source_2_middle_graph1, source_2_middle_graph1_v5,
                                   target_2_middle_graph1, target_2_middle_graph1_v5), dim=-1)
        middle_graph1 = self.fc_graph_middle.forward(middle_graph1, relu=True)


        ### seconde task
        source_graph2 = self.source_graph_conv1.forward(source_graph1, adj=adj2_source, relu=True)
        target_graph2 = self.target_graph_conv1.forward(target_graph1, adj=adj1_target, relu=True)
        middle_graph2 = self.target_graph_conv1.forward(middle_graph1, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph2_v5 = self.transpose_graph_source2target.forward(source_graph2, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph2_v5 = self.transpose_graph_source2middle.forward(source_graph2, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph2_v5 = self.transpose_graph_target2source.forward(target_graph2, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph2_v5 = self.transpose_graph_target2middle.forward(target_graph2, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph2_v5 = self.transpose_graph_middle2source.forward(middle_graph2, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph2_v5 = self.transpose_graph_middle2target.forward(middle_graph2, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph2 = self.similarity_trans(source_graph2, target_graph2)
        source_2_middle_graph2 = self.similarity_trans(source_graph2, middle_graph2)
        # target 2 source middle
        target_2_source_graph2 = self.similarity_trans(target_graph2, source_graph2)
        target_2_middle_graph2 = self.similarity_trans(target_graph2, middle_graph2)
        # middle 2 source target
        middle_2_source_graph2 = self.similarity_trans(middle_graph2, source_graph2)
        middle_2_target_graph2 = self.similarity_trans(middle_graph2, target_graph2)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph2 = torch.cat(
            (source_graph2, target_2_source_graph2, target_2_source_graph2_v5,
             middle_2_source_graph2, middle_2_source_graph2_v5), dim=-1)
        source_graph2 = self.fc_graph_source.forward(source_graph2, relu=True)
        # target
        target_graph2 = torch.cat(
            (target_graph2, source_2_target_graph2, source_2_target_graph2_v5,
             middle_2_target_graph2, middle_2_target_graph2_v5), dim=-1)
        target_graph2 = self.fc_graph_target.forward(target_graph2, relu=True)
        # middle
        middle_graph2 = torch.cat((middle_graph2, source_2_middle_graph2, source_2_middle_graph2_v5,
                                   target_2_middle_graph2, target_2_middle_graph2_v5), dim=-1)
        middle_graph2 = self.fc_graph_middle.forward(middle_graph2, relu=True)


        ### third task
        source_graph3 = self.source_graph_conv1.forward(source_graph2, adj=adj2_source, relu=True)
        target_graph3 = self.target_graph_conv1.forward(target_graph2, adj=adj1_target, relu=True)
        middle_graph3 = self.target_graph_conv1.forward(middle_graph2, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph3_v5 = self.transpose_graph_source2target.forward(source_graph3, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph3_v5 = self.transpose_graph_source2middle.forward(source_graph3, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph3_v5 = self.transpose_graph_target2source.forward(target_graph3, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph3_v5 = self.transpose_graph_target2middle.forward(target_graph3, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph3_v5 = self.transpose_graph_middle2source.forward(middle_graph3, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph3_v5 = self.transpose_graph_middle2target.forward(middle_graph3, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph3 = self.similarity_trans(source_graph3, target_graph3)
        source_2_middle_graph3 = self.similarity_trans(source_graph3, middle_graph3)
        # target 2 source middle
        target_2_source_graph3 = self.similarity_trans(target_graph3, source_graph3)
        target_2_middle_graph3 = self.similarity_trans(target_graph3, middle_graph3)
        # middle 2 source target
        middle_2_source_graph3 = self.similarity_trans(middle_graph3, source_graph3)
        middle_2_target_graph3 = self.similarity_trans(middle_graph3, target_graph3)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph3 = torch.cat(
            (source_graph3, target_2_source_graph3, target_2_source_graph3_v5,
             middle_2_source_graph3, middle_2_source_graph3_v5), dim=-1)
        source_graph3 = self.fc_graph_source.forward(source_graph3, relu=True)
        # target
        target_graph3 = torch.cat(
            (target_graph3, source_2_target_graph3, source_2_target_graph3_v5,
             middle_2_target_graph3, middle_2_target_graph3_v5), dim=-1)
        target_graph3 = self.fc_graph_target.forward(target_graph3, relu=True)
        # middle
        middle_graph3 = torch.cat((middle_graph3, source_2_middle_graph3, source_2_middle_graph3_v5,
                                   target_2_middle_graph3, target_2_middle_graph3_v5), dim=-1)
        middle_graph3 = self.fc_graph_middle.forward(middle_graph3, relu=True)

        return source_graph3, target_graph3, middle_graph3, x

    def similarity_trans(self,source,target):
        sim = torch.matmul(F.normalize(target, p=2, dim=-1), F.normalize(source, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def bottom_forward_source(self, input, source_graph):
        # print('input size')
        # print(input.size())
        # print(source_graph.size())
        graph = self.source_graph_2_fea.forward(source_graph, input)
        x = self.source_skip_conv(input)
        x = x + graph
        x = self.source_semantic(x)
        return x

    def bottom_forward_target(self, input, target_graph):
        graph = self.target_graph_2_fea.forward(target_graph, input)
        x = self.target_skip_conv(input)
        x = x + graph
        x = self.semantic(x)
        return x

    def bottom_forward_middle(self, input, target_graph):
        graph = self.middle_graph_2_fea.forward(target_graph, input)
        x = self.middle_skip_conv(input)
        x = x + graph
        x = self.middle_semantic(x)
        return x

    def forward(self, input_source, input_target=None, input_middle=None, adj1_target=None, adj2_source=None,
                adj3_transfer_s2t=None, adj3_transfer_t2s=None, adj4_middle=None,adj5_transfer_s2m=None,
                adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):
        if input_source is None and input_target is not None and input_middle is None:
            # target
            target_batch = input_target.size(0)
            input = input_target

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target, adj2_source=adj2_source,
                                                             adj3_transfer_s2t=adj3_transfer_s2t,
                                                             adj3_transfer_t2s=adj3_transfer_t2s,
                                                           adj4_middle=adj4_middle,
                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            # source_x = self.bottom_forward_source(source_x, source_graph)
            target_x = self.bottom_forward_target(x, target_graph)

            target_x = F.upsample(target_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, target_x, None

        if input_source is not None and input_target is None and input_middle is None:
            # source
            source_batch = input_source.size(0)
            source_list = range(source_batch)
            input = input_source

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            source_x = self.bottom_forward_source(x, source_graph)
            source_x = F.upsample(source_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return source_x, None, None

        if input_middle is not None and input_source is None and input_target is None:
            # middle
            input = input_middle

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            middle_x = self.bottom_forward_middle(x, source_graph)
            middle_x = F.upsample(middle_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, None, middle_x


class deeplab_xception_end2end_3d_synbn(deeplab_xception_transfer_basemodel_savememory_synbn):
    def __init__(self, nInputChannels=3, n_classes=20, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=7, middle_classes=18, transfer_graph=None):
        super(deeplab_xception_end2end_3d_synbn, self).__init__(nInputChannels=nInputChannels,
                                                                n_classes=n_classes,
                                                                os=os, )
        ### source graph
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=source_classes)
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.source_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=source_classes
                                                                   )
        self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])
        self.source_semantic = nn.Conv2d(out_channels,source_classes,1)
        self.middle_semantic = nn.Conv2d(out_channels, middle_classes, 1)

        ### target graph 1
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### middle
        self.middle_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=middle_classes)
        self.middle_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.middle_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.middle_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### multi transpose
        self.transpose_graph_source2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=n_classes)
        self.transpose_graph_target2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=source_classes)

        self.transpose_graph_middle2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)
        self.transpose_graph_middle2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)

        self.transpose_graph_source2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=middle_classes)
        self.transpose_graph_target2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=middle_classes)


        self.fc_graph_source = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_target = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_middle = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)


    def top_forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer_s2t=None, adj3_transfer_t2s=None,
            adj4_middle=None,adj5_transfer_s2m=None,adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)
        ### target source
        target_graph = self.target_featuremap_2_graph(x)
        ### middle source
        middle_graph = self.middle_featuremap_2_graph(x)

        ##### end2end multi task

        ### first task
        # print(source_graph.size(),target_graph.size())
        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        target_graph1 = self.target_graph_conv1.forward(target_graph, adj=adj1_target, relu=True)
        middle_graph1 = self.target_graph_conv1.forward(middle_graph, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph1_v5 = self.transpose_graph_source2target.forward(source_graph1, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph1_v5 = self.transpose_graph_source2middle.forward(source_graph1,adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph1_v5 = self.transpose_graph_target2source.forward(target_graph1, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph1_v5 = self.transpose_graph_target2middle.forward(target_graph1, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph1_v5 = self.transpose_graph_middle2source.forward(middle_graph1, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph1_v5 = self.transpose_graph_middle2target.forward(middle_graph1, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph1 = self.similarity_trans(source_graph1, target_graph1)
        source_2_middle_graph1 = self.similarity_trans(source_graph1, middle_graph1)
        # target 2 source middle
        target_2_source_graph1 = self.similarity_trans(target_graph1, source_graph1)
        target_2_middle_graph1 = self.similarity_trans(target_graph1, middle_graph1)
        # middle 2 source target
        middle_2_source_graph1 = self.similarity_trans(middle_graph1, source_graph1)
        middle_2_target_graph1 = self.similarity_trans(middle_graph1, target_graph1)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph1 = torch.cat(
            (source_graph1, target_2_source_graph1, target_2_source_graph1_v5,
             middle_2_source_graph1, middle_2_source_graph1_v5), dim=-1)
        source_graph1 = self.fc_graph_source.forward(source_graph1, relu=True)
        # target
        target_graph1 = torch.cat(
            (target_graph1, source_2_target_graph1, source_2_target_graph1_v5,
             middle_2_target_graph1, middle_2_target_graph1_v5), dim=-1)
        target_graph1 = self.fc_graph_target.forward(target_graph1, relu=True)
        # middle
        middle_graph1 = torch.cat((middle_graph1, source_2_middle_graph1, source_2_middle_graph1_v5,
                                   target_2_middle_graph1, target_2_middle_graph1_v5), dim=-1)
        middle_graph1 = self.fc_graph_middle.forward(middle_graph1, relu=True)


        ### seconde task
        source_graph2 = self.source_graph_conv1.forward(source_graph1, adj=adj2_source, relu=True)
        target_graph2 = self.target_graph_conv1.forward(target_graph1, adj=adj1_target, relu=True)
        middle_graph2 = self.target_graph_conv1.forward(middle_graph1, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph2_v5 = self.transpose_graph_source2target.forward(source_graph2, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph2_v5 = self.transpose_graph_source2middle.forward(source_graph2, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph2_v5 = self.transpose_graph_target2source.forward(target_graph2, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph2_v5 = self.transpose_graph_target2middle.forward(target_graph2, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph2_v5 = self.transpose_graph_middle2source.forward(middle_graph2, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph2_v5 = self.transpose_graph_middle2target.forward(middle_graph2, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph2 = self.similarity_trans(source_graph2, target_graph2)
        source_2_middle_graph2 = self.similarity_trans(source_graph2, middle_graph2)
        # target 2 source middle
        target_2_source_graph2 = self.similarity_trans(target_graph2, source_graph2)
        target_2_middle_graph2 = self.similarity_trans(target_graph2, middle_graph2)
        # middle 2 source target
        middle_2_source_graph2 = self.similarity_trans(middle_graph2, source_graph2)
        middle_2_target_graph2 = self.similarity_trans(middle_graph2, target_graph2)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph2 = torch.cat(
            (source_graph2, target_2_source_graph2, target_2_source_graph2_v5,
             middle_2_source_graph2, middle_2_source_graph2_v5), dim=-1)
        source_graph2 = self.fc_graph_source.forward(source_graph2, relu=True)
        # target
        target_graph2 = torch.cat(
            (target_graph2, source_2_target_graph2, source_2_target_graph2_v5,
             middle_2_target_graph2, middle_2_target_graph2_v5), dim=-1)
        target_graph2 = self.fc_graph_target.forward(target_graph2, relu=True)
        # middle
        middle_graph2 = torch.cat((middle_graph2, source_2_middle_graph2, source_2_middle_graph2_v5,
                                   target_2_middle_graph2, target_2_middle_graph2_v5), dim=-1)
        middle_graph2 = self.fc_graph_middle.forward(middle_graph2, relu=True)


        ### third task
        source_graph3 = self.source_graph_conv1.forward(source_graph2, adj=adj2_source, relu=True)
        target_graph3 = self.target_graph_conv1.forward(target_graph2, adj=adj1_target, relu=True)
        middle_graph3 = self.target_graph_conv1.forward(middle_graph2, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph3_v5 = self.transpose_graph_source2target.forward(source_graph3, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph3_v5 = self.transpose_graph_source2middle.forward(source_graph3, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph3_v5 = self.transpose_graph_target2source.forward(target_graph3, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph3_v5 = self.transpose_graph_target2middle.forward(target_graph3, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph3_v5 = self.transpose_graph_middle2source.forward(middle_graph3, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph3_v5 = self.transpose_graph_middle2target.forward(middle_graph3, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph3 = self.similarity_trans(source_graph3, target_graph3)
        source_2_middle_graph3 = self.similarity_trans(source_graph3, middle_graph3)
        # target 2 source middle
        target_2_source_graph3 = self.similarity_trans(target_graph3, source_graph3)
        target_2_middle_graph3 = self.similarity_trans(target_graph3, middle_graph3)
        # middle 2 source target
        middle_2_source_graph3 = self.similarity_trans(middle_graph3, source_graph3)
        middle_2_target_graph3 = self.similarity_trans(middle_graph3, target_graph3)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph3 = torch.cat(
            (source_graph3, target_2_source_graph3, target_2_source_graph3_v5,
             middle_2_source_graph3, middle_2_source_graph3_v5), dim=-1)
        source_graph3 = self.fc_graph_source.forward(source_graph3, relu=True)
        # target
        target_graph3 = torch.cat(
            (target_graph3, source_2_target_graph3, source_2_target_graph3_v5,
             middle_2_target_graph3, middle_2_target_graph3_v5), dim=-1)
        target_graph3 = self.fc_graph_target.forward(target_graph3, relu=True)
        # middle
        middle_graph3 = torch.cat((middle_graph3, source_2_middle_graph3, source_2_middle_graph3_v5,
                                   target_2_middle_graph3, target_2_middle_graph3_v5), dim=-1)
        middle_graph3 = self.fc_graph_middle.forward(middle_graph3, relu=True)

        return source_graph3, target_graph3, middle_graph3, x

    def similarity_trans(self,source,target):
        sim = torch.matmul(F.normalize(target, p=2, dim=-1), F.normalize(source, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def bottom_forward_source(self, input, source_graph):
        # print('input size')
        # print(input.size())
        # print(source_graph.size())
        graph = self.source_graph_2_fea.forward(source_graph, input)
        x = self.source_skip_conv(input)
        x = x + graph
        x = self.source_semantic(x)
        return x

    def bottom_forward_target(self, input, target_graph):
        graph = self.target_graph_2_fea.forward(target_graph, input)
        x = self.target_skip_conv(input)
        x = x + graph
        x = self.semantic(x)
        return x

    def bottom_forward_middle(self, input, target_graph):
        graph = self.middle_graph_2_fea.forward(target_graph, input)
        x = self.middle_skip_conv(input)
        x = x + graph
        x = self.middle_semantic(x)
        return x

    def forward(self, input_source, input_target=None, input_middle=None, adj1_target=None, adj2_source=None,
                adj3_transfer_s2t=None, adj3_transfer_t2s=None, adj4_middle=None,adj5_transfer_s2m=None,
                adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):

        if input_source is None and input_target is not None and input_middle is None:
            # target
            target_batch = input_target.size(0)
            input = input_target

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target, adj2_source=adj2_source,
                                                             adj3_transfer_s2t=adj3_transfer_s2t,
                                                             adj3_transfer_t2s=adj3_transfer_t2s,
                                                           adj4_middle=adj4_middle,
                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            # source_x = self.bottom_forward_source(source_x, source_graph)
            target_x = self.bottom_forward_target(x, target_graph)

            target_x = F.upsample(target_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, target_x, None

        if input_source is not None and input_target is None and input_middle is None:
            # source
            source_batch = input_source.size(0)
            source_list = range(source_batch)
            input = input_source

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            source_x = self.bottom_forward_source(x, source_graph)
            source_x = F.upsample(source_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return source_x, None, None

        if input_middle is not None and input_source is None and input_target is None:
            # middle
            input = input_middle

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            middle_x = self.bottom_forward_middle(x, source_graph)
            middle_x = F.upsample(middle_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, None, middle_x


if __name__ == '__main__':
    net = deeplab_xception_end2end_3d()
    net.freeze_totally_bn()
    img1 = torch.rand((1,3,128,128))
    img2 = torch.rand((1, 3, 128, 128))
    a1 = torch.ones((1,1,7,20))
    a2 = torch.ones((1,1,20,7))
    net.eval()
    net.forward(img1,img2,adj3_transfer_t2s=a2,adj3_transfer_s2t=a1)