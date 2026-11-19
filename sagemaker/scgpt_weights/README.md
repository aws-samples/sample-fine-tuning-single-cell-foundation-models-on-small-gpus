# scGPT pretrained weights (build-context only)

Stage each preset's checkpoint triple here before building `Dockerfile.scgpt`:

    sagemaker/scgpt_weights/scGPT_human/{model.pt,args.json,vocab.json}
    sagemaker/scgpt_weights/scGPT_bc/{model.pt,args.json,vocab.json}

Weights are large and are gitignored (only this README + .gitkeep are tracked).
Download them from the scGPT release (https://github.com/bowang-lab/scGPT).
