# This example requires the 'message_content' intent.
import os
import random

import discord
import logging

from discord import channel

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

    if str(message.attachments) == "[]":  # Checks if there is an attachment on the message
        return
    else:  # If there is it gets the filename from message.attachments
        split_v1 = str(message.attachments).split("filename='")[1]
        filename = str(split_v1).split("' ")[0]
        if filename.endswith(".jpg"):
            await message.attachments[0].save(fp="SaveImage/{}".format(filename))
            await message.channel.send('Get a jpg file.')
            print("Get a jpg file.")
        if filename.endswith(".png"):
            await message.attachments[0].save(fp="SaveImage/{}".format(filename))
            await message.channel.send('Get a png file.')
            print("Get a png file.")




client.run('TOKEN',log_handler=handler)
