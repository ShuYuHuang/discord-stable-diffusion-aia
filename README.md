# AIA bot 1 - Discord Bot for Stable Diffuser!

## Setup
1. Clone the repo, then install the dependencies in ``requirements.sh``
2. In ``env.txt`` include your HuggingFace token and the Discord token for your bot
3. Simply change the name of ``env.txt`` to ``.env`` with the following command (please remember to neglect the $ in the beginning)
```$ mv env.txt .env```
4. Then simply execute the script. Or you can follow the steps in host_command.txt to execute that in background

```$ python bot.py```

### Quickstart
#### Text to Image

To generate an image from text, use the ``!dream`` command and include your prompt as the query. Enjoy

"!dream A portrait of a white bunny resting under moonlight , a highly intricate and hyperdetailed matte painting by Huang Guangjian, Anna Dittmann and Dan Witz, fantasy art, album cover art, celestial"
![image](https://cdn.discordapp.com/attachments/1015428907379462154/1017271908208738344/0.png)

## Citation
```
@InProceedings{Rombach_2022_CVPR,
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
  title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  pages     = {10684-10695}
}
```
The method of using diffuser model is taken from:
```
https://github.com/huggingface/diffusers
https://github.com/replicate/cog-stable-diffusion
```
The method of deploying discord bot is taken from:
```
https://github.com/harubaru/discord-stable-diffusion
```