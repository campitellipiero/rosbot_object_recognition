from setuptools import setup

package_name = 'rosbot_object_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuctella',
    maintainer_email='nuctella@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "obj_rec = rosbot_object_recognition.object_recognition.object_recognition_clean_withorientation:main"
        
        ],
    },
)
