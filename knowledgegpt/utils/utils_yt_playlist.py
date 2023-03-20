def yt_playlist_link_parser(playlist_link:str):
    import yt_dlp
    ydl_opts = {
        'dump_single_json': True,
        'extract_flat': True,
        'skip_download': True,
        'playlist_items': '1-1000', 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(playlist_link, download=False)
        video_list = playlist_info['entries']
        video_links = [f"https://www.youtube.com/watch?v={video['id']}" for video in video_list]

    return video_links