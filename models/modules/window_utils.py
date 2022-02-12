def window_partition(x, window_size):
    """
    将feature map按照window_size分割成windows
    """

    b, h, w, c = x.shape
    # (b,h,w,c) -> (b,h//m,m,w//m,m,c)
    x = x.view(b, h//window_size, window_size, w//window_size, window_size, c)
    # (b,h//m,m,w//m,m,c) -> (b,h//m,w//m,m,m,c)
    # -> (b,h//m*w//m,m,m,c) -> (b*n_windows,m,m,c)
    windows = (x
               .permute(0, 1, 3, 2, 4, 5)
               .contiguous()
               .view(-1, window_size, window_size, c))
    return windows

def window_reverse(x,window_size,h,w):
    """
    将分割后的windows还原成feature map
    """

    b = int(x.shape[0] / (h*w/window_size/window_size))
    # (b,h//m,w//m,m,m,c)
    x = x.view(b,h//window_size,w//window_size,window_size,window_size,-1)
    # (b,h//m,w//m,m,m,c) -> (b,h//m,m,w//m,m,c)
    # -> (b,h,w,c)
    x = x.permute(0,1,3,2,4,5).contiguous().view(b,h,w,-1)

    return x
