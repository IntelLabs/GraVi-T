from pkg_resources import get_distribution

try:
    __version__ = get_distribution('gravit').version
except:
    __version__ = '1.1.0'
