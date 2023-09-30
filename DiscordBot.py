# This example requires the 'message_content' intent.
import Model_Apply
import os
import discord
import logging
import time

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    localtime = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    print(f'We have logged in as {client.user} at {localtime}')


@client.event
async def on_message(message):
    localtime = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    if message.author == client.user:
        return

    if message.content.startswith('$status'):
        if message.author.id == 495213806931279873 or 445156059099693056:
            tmp = message.content.split(" ", 8)
            if len(tmp) == 1:
                await message.channel.send('nothing be changed')
            else:
                status = tmp[1]
                print(f"status set as {status}")
                act = discord.CustomActivity(status)
                await client.change_presence(activity=act)
        else:
            await message.channel.send('沒有權限')

    if message.channel.type is discord.ChannelType.private:
        print(f" {message.author.global_name} say：{message.content}")

    if message.content.startswith('$stop'):
        if message.author.id == 495213806931279873 or 445156059099693056:
            print(exit)
            exit()
        else:
            await message.channel.send('住手！')

    if message.content.startswith('$restart'):
        if message.author.id == 495213806931279873 or 445156059099693056:
            os.system('DiscordBot.bat')
        else:
            await message.channel.send('住手！')

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

    if message.content.startswith('中秋'):
        if message.channel.type is discord.ChannelType.private:
            await message.channel.send('中秋節快樂！接下來一起加油吧OuO')
        else:
            return

    if str(message.attachments) == "[]":  # Checks if there is an attachment on the message
        return
    else:  # If there is it gets the filename from message.attachments
        split_v1 = str(message.attachments).split("filename='")[1]
        filename = str(split_v1).split("' ")[0]
        if filename.endswith('.jpg'):
            await message.attachments[0].save(fp="E:/SaveImage/{}".format(filename))
            result = Model_Apply.predict_external_image("E:/SaveImage/{}".format(filename))
            await message.channel.send(f'謝謝 {message.author.global_name} 投餵了一個 {result} ！')
            print(f"Get a jpg file at {localtime} from {message.author.global_name}.")
        if filename.endswith(".png"):
            await message.attachments[0].save(fp="E:/SaveImage/{}".format(filename))
            result = Model_Apply.predict_external_image("E:/SaveImage/{}".format(filename))
            await message.channel.send(f'謝謝 {message.author.global_name} 投餵了一個 {result} ！')
            print(f"Get a png file at {localtime} from {message.author.global_name}.")
        if filename.endswith(".mp4"):
            await message.attachments[0].save(fp="E:/SaveVideo/{}".format(filename))
            await message.channel.send(f'{message.author.global_name} 太...太多了')
            print(f"Get a mp4 file at {localtime} from {message.author.global_name}.")


client.run('TOKEN', log_handler=handler)
