from setuptools import setup, find_packages

setup(
    name="leatherback_v0",  # Corresponds to the package name in source/
    version="0.1.0",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        # Add other direct dependencies if any
    ],
    author="Your Name", # Replace with your name
    author_email="your.email@example.com", # Replace with your email
    description="A Leatherback robot driving environment for Isaac Lab",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MuammerBay/isaac-leatherback", # Replace with your repo URL if different
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License", # Assuming BSD-3-Clause based on Isaac Lab
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    package_data={
        # Adjust 'leatherback_v0' if your package name is different
        "Leatherback": ["tasks/direct/leatherback/custom_assets/*.usd"],
    },
    zip_safe=False,
) 