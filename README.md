## About the project
The objective of this project is to use Categorical Naive Bayes Classification Algorithm to determine the species & venomous status of a particular snake given various attributes.
The dataset provides classifying features for the following snake species (along with their venomous statuses):
  - Coral Snake
  - Copperhead
  - Cottonmouth
  - Glossy Snake
  - Eastern Ratsnake
  - Wormsnake
  - Rubber Boa
  - Diamondback Rattlesnake
  - Dessert Sidewinder
  - Fox Snake

The features that are used to classify each snake species along with the various categories are as follows:
  - Body shape -- (slender, stout, typical, unknown)
  - Head shape -- (round, broad, pointed, triangular, unknown)
  - Color -- (red/black/yellow, tan, brown, grey, black, black/yellow)
  - Back pattern -- (striped, banded, blotched, none, unknown)
  - Scale texture -- (smooth, keeled, weakly keeled, unknown)
  - Eye pupil shape -- (round, vertical, unknown)

### Built With
  - Python 3.12.2
  - NumPy 1.26.4
  - Pandas 2.2.1
  - openpyxl 3.1.2


## Getting Started
This is a step-by-step guide to setting up the project on your local machine.

### Prerequisites
- NumPy 1.26.4
    `pip install numpy==1.26.4`
- Pandas 2.2.1
    `pip install pandas==2.2.1`
- openpyxl 3.1.2
    `pip install openpyxl==3.1.2`

### Installation
1. Clone the repository
  HTTPS : `git clone https://github.com/Str-Josh/SnakeSpeciesClassification.git`
  SSH : `git@github.com:Str-Josh/SnakeSpeciesClassification.git`
3. Change git remote url to avoid accidental pushes to main project
  `git remote set-url origin github_username/repository_name
   git remote -v # Confirms the changes`


## Usage

### Classifying a snake given particular attributes
1. Within the entry point of the code ( if __name__ == "__main__": ), there's be a line `classifier = Classifier(example_input1, naive_bayes_table, data.classifications)`.
2. In this line, replace `example_input1` with a list of the features of your snake you want to classify.
3. Execute the python script.





