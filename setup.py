import setuptools

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith(("#", "-r"))
    ]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="sciagenttrace",
    version="1.0.0",
    description=(
        "Protocol execution and trace processing for SciAgentTrace, a matched "
        "multi-agent scientific-reasoning trace corpus."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        include=[
            "agentverse", "agentverse.*",
            "agentverse_command", "agentverse_command.*",
            "dataloader", "dataloader.*",
        ]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest"],
        "omni-judge": ["transformers>=4.48,<5", "accelerate", "torch"],
    },
    entry_points={
        "console_scripts": [
            "agentverse-benchmark = agentverse_command.benchmark:cli_main",
            "agentverse-tasksolving = agentverse_command.main_tasksolving_cli:cli_main",
            "agentverse-trace-analysis = agentverse_command.trace_analysis:cli_main",
            "agentverse-trace-diagnostics = agentverse_command.trace_diagnostics:cli_main",
            "agentverse-rq1-outcome-cost = agentverse_command.rq1_outcome_cost:cli_main",
        ],
    },
)
