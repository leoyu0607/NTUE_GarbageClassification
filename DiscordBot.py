import Model_Apply
import os
import discord
import logging
import time

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

adminID = [495213806931279873, 445156059099693056]
# 圖片副檔名list
filename_list = ['.jpg', '.jpeg', 'png']


@client.event
async def on_ready():
    localtime = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    print(f'We have logged in as {client.user} at {localtime}')


@client.event
async def on_message(message):
    localtime = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    if message.author == client.user:
        return

    if message.channel.type is discord.ChannelType.private:
        print(f"[{localtime}] {message.author.global_name} say：{message.content}")

    if message.content.startswith('$status'):
        if message.author.id in adminID:
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

    if message.content.startswith('$stop'):
        if message.author.id in adminID:
            print(exit)
            exit()
        else:
            await message.channel.send('住手！')

    if message.content.startswith('$restart'):
        if message.author.id in adminID:
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

    if message.channel.type is discord.ChannelType.private:
        if str(message.attachments) == "[]":  # Checks if there is an attachment on the message
            return
        else:  # If there is it gets the filename from message.attachments
            split_v1 = str(message.attachments).split("filename='")[1]
            filename = str(split_v1).split("' ")[0]
            for i in range(0, 3):
                if filename.endswith(filename_list[i]):
                    await message.attachments[0].save(fp="SaveImage/{}".format(filename))
                    result = Model_Apply.predict_result("SaveImage/{}".format(filename))
                    days = Model_Apply.days(result)
                    await message.channel.send(f'{message.author.global_name} ，這是一個 {result}。')
                    if days != '不可回收':
                        await message.channel.send(f'建議在每週 {days} 回收！')
                    else:
                        await message.channel.send(f'{days}！')
                    print(f"Get a {filename_list[i]} file at {localtime} from {message.author.global_name}.")


client.run('TOKEN', log_handler=handler)
