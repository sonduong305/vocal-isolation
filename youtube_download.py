import youtube_dl


ydl_opts = {
    'format': 'bestaudio/mp3',
    'outtmpl': 'temp.webm',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '256',
        'r': '44100',
    }],
}
def get_audio(link):
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        # ydl.download([link])
        return ydl.extract_info(link, download=True).get('title', None)