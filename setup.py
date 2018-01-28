from setuptools import setup, find_packages

setup(
    name='finding_simple_lane_lines',
    version='0.1',
    author='afomin',
    company='afomin',
    author_email='alexanderfomin1992@mail.ru',
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'finding_simple_lane_lines= finding_simple_lane_lines.finding_simple_lane_lines:main',
        ],
    },
    package_data={
        'finding_simple_lane_lines': ['test_images/*', ],
        'finding_simple_lane_lines': ['test_videos/*', ]
    },
)
