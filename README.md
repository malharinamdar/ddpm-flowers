# DDPM-Flowers

Experiments on the DDPM from [Neelesh-Gupta/DDPM-Flowers](https://github.com/Neelesh-Gupta/DDPM-Flowers) on the
Kaggle [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) dataset (4,317 photos,
5 classes, resized to 32×32 as in the original). The goal was a lower FID than the original model, using only free
Kaggle GPUs (Tesla T4). Every notebook here was run on Kaggle; the outputs it produced are in `outputs/`.

## Results

All numbers compare generated images with all 4,317 real images at 32×32. FID@2k uses 2,000 generated images
(mean ± std over 5 separate sets where shown). Lower is better for FID and KID.

| Model | Sampler | Params | FID@2k | KID@2k | FID@10k | Precision / Recall |
|---|---|---|---|---|---|---|
| Original notebook (20 epochs) | original, 999 steps | 25.3M | 248.5 | 0.168 | – | – / 0.02 |
| Original notebook (20 epochs) | clipped, 250 steps | 25.3M | 172.2 | 0.126 | – | – |
| Original architecture, retrained 200 epochs | original, 999 steps | 25.3M | 56.8 | 0.0285 | – | – |
| Pretrained DDPM, fine-tuned 300 epochs | original, 999 steps | 35.7M | 35.2 ± 1.0 | 0.0136 | 25.7 | 0.78 / 0.50 |
| **Pretrained DDPM, fine-tuned 400 epochs** | **clipped, 250 steps** | 35.7M | **27.2 ± 0.4** | **0.0059** | **17.7** | **0.81 / 0.52** |
| Stable Diffusion v1.5 + LoRA | DPM-Solver++, 25 steps | 860M + 6.4M | 33.1 ± 0.4 | 0.0083 | 23.2 | 0.71 / 0.56 |
| Real images vs real images | – | – | 22.6 | ≈ 0 | – | – |

The last row is a reference point: two halves of the real dataset (2,158 images each) compared with each other.
FID stays well above zero even between two sets of real images of this size, and FID values change with the number
of images used, so they should only be compared when measured the same way.

## What was done

1. **Reran the original notebook unchanged** and scored it. Its published run had turned to NaN (fp16 mixed precision);
   our rerun trained normally and gives FID@2k 248.5.
2. **Retrained the same architecture with a standard recipe**: EMA of the weights, gradient clipping, learning-rate
   warm-up and 200 epochs instead of 20. FID@2k 56.8, so the original model was mostly under-trained.
3. **Fine-tuned a pretrained DDPM** ([google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32), trained
   on CIFAR-10, same image size and noise schedule) with the same recipe. FID@2k 35.2 at 300 epochs.
4. **Changed the sampler**: the predicted clean image is clipped to [-1, 1] at every step, the last step goes to t = 0,
   and a DDIM-style update with 250 steps is used. FID@2k 35.2 → 27.3 with no retraining (and 248.5 → 172.2 for the
   original model). Sampling is also about 4× faster.
5. **Trained longer and picked the checkpoint**: FID levels off around 300–400 epochs and gets worse at 500–600
   (overfitting), so epoch 400 is used for the final evaluation on 10,000 images.
6. **Compared with a much larger model**: Stable Diffusion v1.5 with LoRA adapters, trained on the flowers at 256×256
   and downsampled to 32×32 for scoring. It gives sharper, more varied images but a worse FID (33.1).

## Notebooks (in the order they were run)

| Folder / notebook | What it does |
|---|---|
| `1_reference_model/01_reference_notebook_rerun.ipynb` | The original notebook, run unchanged (20 epochs). Last cell generates 2,000 images from the epoch-20 checkpoint with the original sampler. |
| `1_reference_model/02_score_reference.ipynb` | Scores those 2,000 images (FID, KID) and computes the real-vs-real floor. |
| `1_reference_model/03_reference_retrained_200ep.ipynb` | Same architecture, retrained for 200 epochs with EMA, clipping and warm-up; scored with the original sampler. |
| `2_pretrained_ddpm/01_finetune_300ep.ipynb` | Fine-tunes the pretrained CIFAR-10 DDPM on the flowers for 300 epochs; saves checkpoints every 25 epochs. |
| `2_pretrained_ddpm/02_eval_reference_sampler.ipynb` | Scores the epoch-300 model with the original 999-step sampler on 10,000 images (all metrics). |
| `2_pretrained_ddpm/03_sampler_sweep.ipynb` | Tries the new sampler with 100/250 steps and η = 0/1, then compares early checkpoints. |
| `2_pretrained_ddpm/04_continue_training_and_final_eval.ipynb` | Continues training to 600 epochs, compares checkpoints 300–600, evaluates epoch 400 on 10,000 images, and scores the original model with the same metrics. |
| `3_stable_diffusion_lora/sd15_lora_train_and_eval.ipynb` | Stable Diffusion v1.5 + LoRA: training, guidance sweep (1.0 / 1.5 / 2.0) and the same evaluation. |

`outputs/` holds the result files, training logs and sample grids from these runs, in matching folders.

## How the evaluation works

- **FID@2k** (main comparison): 2,000 generated images vs all 4,317 real images, same code for every model
  (`torchmetrics`, standard Inception weights).
- **KID**: like FID but unbiased with few images.
- **FID at 2k–10k**: FID drops as more generated images are used, so we also report FID at 4k–10k generated images.
- **Precision / recall**: how realistic the images are (precision) and how much of the real variety they cover (recall).
- **Inception Score**: reported for completeness; it is weak for flowers (real images only score 6.29).
- **Memorisation check**: the closest training image to each of 64 generated images, compared with how close real
  images are to each other. No copying was found.
- Checkpoints and sampler settings were chosen on one set of 2,000 images; the final 10,000 images use new random noise.

## Running the notebooks

1. On Kaggle, import a notebook (File → Import Notebook).
2. Add the **Flowers Recognition** dataset as input. Notebooks that use earlier results (e.g. `02_score_reference`,
   `02_eval_reference_sampler`, `04_continue_training_and_final_eval`) also need the earlier notebook's output added
   as input.
3. Settings: **GPU T4 ×2**, internet on.
4. **Save Version → Save & Run All**. Results are written to `/kaggle/working/`.

## Notes

- Code comments and the experiment options that were not used in each run were removed for readability; the code is
  otherwise as it was run. Printed outputs are kept.
- Three runs stopped with an error after their main work was saved (a library import order issue, a two-GPU sampling
  issue and a GPU out-of-memory error). Each is marked in the notebook, and the step was redone in the next notebook.
- The original notebook is not copied here; it is in [Neelesh-Gupta/DDPM-Flowers](https://github.com/Neelesh-Gupta/DDPM-Flowers).
  `1_reference_model/01_reference_notebook_rerun.ipynb` is that notebook run unchanged, plus one cell at the end.
- Model weights and the raw generated images are not included because of their size.
- Each setting was trained once (no repeated training seeds), and there was no full hyperparameter search.

## Credits

- Original model and notebook: [Neelesh-Gupta/DDPM-Flowers](https://github.com/Neelesh-Gupta/DDPM-Flowers)
- Dataset: A. Mamaev, *Flowers Recognition* (Kaggle)
- Pretrained models: [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32),
  [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- Methods: DDPM (Ho et al., 2020), DDIM (Song et al., 2021), FID (Heusel et al., 2017), KID (Bińkowski et al., 2018),
  precision/recall (Kynkäänniemi et al., 2019), sample-size effects in FID (Chong & Forsyth, 2020), LoRA (Hu et al., 2022),
  latent diffusion (Rombach et al., 2022)
