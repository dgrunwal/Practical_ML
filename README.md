Book includes ml_demo.py code a workable documented python program demonstrating a real machine learning program.


Qwik Guides — Production ML Reference Set

A set of three self-contained, plain-language reference guides that accompany the Beginning Machine Learning book. Where the book walks a beginner line-by-line through a working scikit-learn project, these guides zoom out to the bigger picture: what it takes to build, ship, and maintain machine learning systems in the real world.
Each guide distills core lessons from Andrew Ng's Machine Learning Engineering for Production (MLOps) specialization (DeepLearning.AI) into a single readable HTML page — no setup, no dependencies. Just open the file in any browser.

The Guides

1. Machine Learning Overview (1_ml_qwik_guide.html)
The end-to-end view. Covers the ML project lifecycle from scoping to deployment, plus concept and data drift, deployment patterns, monitoring a live system, ML pipelines and their cascading effects, and core MLOps tools and principles. Start here for the map of the whole territory.
2. Modeling Challenges & Strategies (2_ml_modeling_challenges_guide.html)
The practitioner's playbook for improving a model. Covers why low average test error isn't enough, precision/recall/F1 on skewed datasets, error analysis, prioritizing what to fix, model-centric vs. data-centric development, data augmentation, feature engineering, experiment tracking, and pre-deployment performance auditing.
3. Data Definition Guide (3_ml_data_definition_guide.html)

The foundation underneath everything else. Covers why defining data is hard, label ambiguity and consistency, human-level performance as a baseline, scoping and feasibility, obtaining data efficiently, reproducible data pipelines, metadata/provenance/lineage, and building balanced train/dev/test splits.

How They Fit the Book

The book teaches the mechanics — train/test splits, feature scaling, training a classifier, reading the results. These guides explain the judgment that surrounds those mechanics in production: how to decide what to work on, how to tell whether a model is actually good, and why clean, well-defined data matters more than a clever algorithm. Read the book to build your first model; keep these nearby as you grow toward shipping real ones.
Format

Each guide is a single standalone HTML file with inline styling — portable, printable, and easy to host. No build step, no external assets.
