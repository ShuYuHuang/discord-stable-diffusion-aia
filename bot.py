## Hugginface Interaction
from huggingface_hub.hf_api import HfApi,HfFolder
from huggingface_hub.commands.user import LogoutCommand
from huggingface_hub._login import login as hf_login

## Model
import torch
from torch.cuda.amp import autocast
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

## Discord interaction
import discord
from discord.ext import commands
from io import BytesIO
# refer to https://discordpy.readthedocs.io/en/latest/ext/commands/commands.html?highlight=discord%20ext%20import%20commands

## OS commands
import os,time
from dotenv import load_dotenv


try:
    ## Load in enviroment variables in .env
    load_dotenv()

    ## log in hugginface
    hf_login(token=os.environ["HF_TOKEN"])
    
    ## Open intent recepter
    intents = discord.Intents.default()
    #intents.message_content = True
    
    print("Start Loading Stable Diffusion Pipeline")
    
    # get config from .env file    
    cfg=dict( 
        num_inference_steps=int(os.environ["INFERENCE_STEPS"]),
        # generator=torch.Generator("cuda").manual_seed(124),
        height=int(os.environ["HEIGHT"]), 
        width=int(os.environ["WIDTH"])
    )
    mydevice=os.environ["DEVICE"]
    ## Initial model
    with torch.cuda.device(mydevice):
        model_id = "stabilityai/stable-diffusion-2-base" # "runwayml/stable-diffusion-v1-5" # "runwayml/stable-diffusion-v1-5" "CompVis/stable-diffusion-v1-4"
        pipe = DiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(mydevice)
    
    ## Setup Chatbot
    bot = commands.Bot(
        command_prefix="!",
        description="Runs diffuser models on AIA hub!",
        intents=intents,
    )
    print("Runs diffuser models on AIA hub!")
    
    @bot.command()
    async def dream(ctx, *, prompt):
        """Generate an image from a text prompt using the stable-diffusion model"""
        
        start_time=time.time()

        ## Temperary message during waiting
        msg = await ctx.send(f"“{prompt}”\n> Generating...",delete_after=1.0)
        
        ## Run model
        with torch.cuda.device(mydevice),autocast():
            images = pipe(prompt, **cfg).images#["sample"]
            
        duration=time.time()-start_time
        
        ## Form return message
        embed = discord.Embed()
        embed.set_footer(text=f"{prompt}\n(Finished in {duration: .02f} seconds)")
        with BytesIO() as buffer:
            images[0].save(buffer, 'PNG')
            buffer.seek(0)
            
            await ctx.send(embed=embed, file=discord.File(fp=buffer, filename='0.png')) 
        
    bot.run(os.environ["DISCORD_TOKEN"])
except :
    ## log out hugginface
    print("Interruped, Log out from Hugginface")
    LogoutCommand(None).run()
    