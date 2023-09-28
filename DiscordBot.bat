@echo off
rem 關閉命令顯示
call C:/Users/20220814/anaconda3/Scripts/activate.bat C:\Users\20220814\anaconda3
rem 開啟Anaconda prompt
call conda activate DiscordBot
rem 激活虛擬環境
call cd C:\Users\20220814\PycharmProjects\NTUE_Garbage_Classification
rem 移動目錄
call python DiscordBot.py