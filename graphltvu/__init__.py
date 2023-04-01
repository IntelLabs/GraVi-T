from pkg_resources import get_distribution

try:
    __version__ = get_distribution('graphltvu').version
except:
    __version__ = '1.0.0'
