"""
Physical layer components for 6G smart factory systems

The PHY classes import TensorFlow and the Sionna PHY. They load on first use, so
that TensorFlow-free tools (the viz image's ray tracing) can import
``factory6g.components.raytracing`` on its own.
"""

__all__ = [
    'AntennaConfig',
    'Transmitter',
    'ChannelModel',
    'Receiver'
]

_LAZY = {
    'AntennaConfig': '.antenna',
    'Transmitter': '.transmitter',
    'ChannelModel': '.channel',
    'Receiver': '.receiver',
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
