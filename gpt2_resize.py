import argparse
import torch

from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import GPT2Model
from transformers import GPT2Tokenizer


def main(args):
    if args.no_pretraining:
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_config(config)
        out_fn = f'models/{args.model}_no_pt_{args.ctx}'
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        out_fn = f'models/{args.model}_{args.ctx}'

    num_position_embeds_diff = args.ctx - model.config.max_position_embeddings
    if num_position_embeds_diff != 0:
        old_position_embeddings_weight = model.transformer.wpe._parameters['weight'].data.clone()
        model.config.max_position_embeddings = args.ctx
        model.transformer.wpe = torch.nn.Embedding(model.config.max_position_embeddings, model.transformer.embed_dim)
        if num_position_embeds_diff > 0:
            model.transformer.wpe.weight.data[:-num_position_embeds_diff] =  torch.nn.Parameter(old_position_embeddings_weight)
        else:
            model.transformer.wpe.weight.data[:] =  torch.nn.Parameter(old_position_embeddings_weight[:num_position_embeds_diff])

    model.save_pretrained(out_fn)
    print(f'resized {args.model} to a context length of {args.ctx} and saved to {out_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gpt2')
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--no_pretraining", type='store_true')

    args = parser.parse_args()
    main(args)