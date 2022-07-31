from setuptools import setup

install_requires = [
    'dalex>=1.4.1'
    
]

setup(name='fairdetect_groupb',
      version='0.47',  # Development release
      description='Library to identify bias in pre-trained models!',
      url='https://github.com/dianisley/fairdetect_b',
      author='GMBD_Group_B',
      author_email='Carlos.BlazquezP@student.ie.edu',
      license='MIT',
          packages=['fairdetect_groupb'],
      zip_safe=False,
      install_requires=install_requires)