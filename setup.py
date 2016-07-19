from setuptools import setup


setup(
    name='xrdpd',
    version='0.1.0',
    author='C. Simpson',
    author_email='c.a.simpson01@gmail.com',
    packages=['xrdpb'],
    include_package_data=True,
    url='https://github.com/casimp/xrdpb',
    download_url = 'https://github.com/casimp/xrdpb/tarball/v0.1.0',
    license='LICENSE.txt',
    description='Create 1D XRD patterns and 2D Debye-Scherrer ringe for simple material systems/diffraction setups.',
    keywords = ['XRD', 'EDXD', 'x-ray', 'diffraction', 'strain', 'synchrotron'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"]
)
