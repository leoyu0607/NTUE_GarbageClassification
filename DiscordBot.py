# This example requires the 'message_content' intent.
import Model_Apply
import os
import random
import discord
import logging
from discord import channel
import time
import datetime

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

status = input("機器人動態：")
print("status set")
act = discord.CustomActivity(status)


@client.event
async def on_ready():
    localtime = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    print(f'We have logged in as {client.user} at {localtime}')
    await client.change_presence(activity=act)


@client.event
async def on_message(message):
    localtime = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    if message.author == client.user:
        return

    if message.content.startswith('$stop'):
        if message.author.id == 495213806931279873:
            print(exit)
            exit()
        else:
            await message.channel.send('閉嘴！')

    if message.content.startswith('$restart'):
        os.system('DiscordBot.bat')

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

    if str(message.attachments) == "[]":  # Checks if there is an attachment on the message
        return
    else:  # If there is it gets the filename from message.attachments
        split_v1 = str(message.attachments).split("filename='")[1]
        filename = str(split_v1).split("' ")[0]
        if filename.endswith('.jpg'):
            await message.attachments[0].save(fp="E:/SaveImage/{}".format(filename))
            await message.channel.send(f'謝謝 {message.author.global_name} 的投喂！')
            print(f"Get a jpg file at {localtime} from {message.author.global_name}.")
        if filename.endswith(".png"):
            await message.attachments[0].save(fp="E:/SaveImage/{}".format(filename))
            await message.channel.send(f'謝謝 {message.author.global_name} 的投喂！')
            print(f"Get a png file at {localtime} from {message.author.global_name}.")
        if filename.endswith(".mp4"):
            await message.attachments[0].save(fp="E:/SaveVideo/{}".format(filename))
            await message.channel.send(f'{message.author.global_name} 太...太多了')
            print(f"Get a mp4 file at {localtime} from {message.author.global_name}.")


client.run('TOKEN',log_handler=handler)
