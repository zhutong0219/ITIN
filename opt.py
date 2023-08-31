import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_name', default='precomp',
                        help='{coco,f30k}_precomp')
parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
parser.add_argument('--finetune', default=False)
parser.add_argument('--cnn_type', default='None')
parser.add_argument('--self_attention', default=False)
parser.add_argument('--cross_model', default=True)
parser.add_argument('--measure', default='gate_fusion_new')
parser.add_argument('--embed_mask', default=False)
parser.add_argument('--finetune_gate', default=True)

opt = parser.parse_args()
