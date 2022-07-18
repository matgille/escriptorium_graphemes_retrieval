Ce script permet de récupérer l'image de chaque graphème d'un manuscrit transcrit *automatiquement* avec eScriptorium (a venir: avec Kraken).

Il utilise l'API d'eScriptorium pour récupérer les informations, les images, et extraires les graphèmes à des fins d'étude et/ou de comparaison.

La proportion de graphèmes conservés permet de prélever des échantillons (option `proportion_to_keep` dans la fichier de paramètres).

Il est nécessaire de disposer de modèles de transcription corrects pour récupérer des images exploitables et fiables. 
Les graphèmes sont classées en reprenant les classes reconnues par le modèle: le script reproduira donc
les biais éventuels du modèle et du jeu de données d'entraînement. Ce script peut servir à identifier ces biais.

Si une zone est préférée, elle sera indiquée dans le fichier configuration. Si le document
n'est pas zoné, marquer ``null``.

## Utilisation

`python3 retrieve_graphemes.py fichier.conf`

Il est nécessaire de lancer une première fois le script avec l'option 
`--identifiers` afin de récupérer les identifiants de transcription et de zone.


La *bounding box* proposée par kraken n'étant pas parfaitement ajustée, un ajustement de la hauteur et de la largeur 
des images extraites est probablement nécessaire. Le fichier de configuration
`pixels_adjustment.json` permet d'ajuster la *box* en fonction d'une classification des graphèmes très simple:
`centre`, `ascendant`, `descendant` (les graphèmes sont classés dans `graphemes_classification.json`). Pour ce faire, on lancera d'abord le script
avec `--classes` pour récupérer les différentes classes de la transcription.
On recommandera de tester différentes configurations avec une seule page pour débuter.
