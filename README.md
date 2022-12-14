Ce script permet de récupérer l'image de chaque graphème d'un manuscrit transcrit *automatiquement* avec eScriptorium (a venir: avec Kraken).

Il utilise l'API d'eScriptorium pour récupérer les informations, les images, et extraires les graphèmes à des fins d'étude et/ou de comparaison.

La proportion de graphèmes conservés permet de prélever des échantillons (option `proportion_to_keep` dans la fichier de paramètres).

Il est nécessaire de disposer de modèles de transcription corrects pour récupérer des images exploitables et fiables. 
Les graphèmes sont classés en reprenant la classification du modèle: le script reproduira donc
les biais éventuels de celui-ci modèle et donc ceux du jeu de données d'entraînement: ce script peut aussi servir à 
identifier ces biais.

Si une zone est préférée, elle sera indiquée dans le fichier configuration. Si le document
ne contient pas de typologie de zones, marquer ``null``.

## Utilisation

Le fichier `example.env` doit être rempli pour contenir les informations de connection à l'API.

`python3 retrieve_graphemes.py fichier.conf`

Il est nécessaire de lancer une première fois le script avec l'option 
`--identifiers` afin de récupérer les identifiants de transcription et de zone, et de
fournir une classification des graphèmes: voir paragraphe suivant.


La *bounding box* calculée à partir des polygones identifiés par kraken n'étant pas parfaitement ajustée, un ajustement de la hauteur et de la largeur 
des images extraites est probablement nécessaire. Le fichier de configuration
`bounding_box_adjustment.json` permet d'ajuster la *bounding box* en fonction d'une classification des graphèmes par rapport à la ligne très simple:
`centre`, `ascendant`, `descendant` (les graphèmes sont classés dans `graphemes_classification.json`). Pour ce faire, on lancera d'abord le script
avec `--classes` pour récupérer les différentes classes de la transcription.
On recommandera de tester différentes configurations avec une seule page pour trouver les réglages adaptés au document.

## Exemples d'images récupérées:


ſ:

![ALT](img/example/graphemes/ſ/46_0.png)
![ALT](img/example/graphemes/ſ/46_10.png)
![ALT](img/example/graphemes/ſ/46_11.png)
![ALT](img/example/graphemes/ſ/46_20.png)
![ALT](img/example/graphemes/ſ/46_40.png)

ꝺ:

![ALT](img/example/graphemes/ꝺ/46_0.png)
![ALT](img/example/graphemes/ꝺ/46_10.png)
![ALT](img/example/graphemes/ꝺ/46_11.png)
![ALT](img/example/graphemes/ꝺ/46_20.png)
![ALT](img/example/graphemes/ꝺ/46_40.png)

h:

![ALT](img/example/graphemes/h/46_0.png)
![ALT](img/example/graphemes/h/46_1.png)
![ALT](img/example/graphemes/h/46_2.png)
![ALT](img/example/graphemes/h/46_3.png)
![ALT](img/example/graphemes/h/46_4.png)

ꝑ:

![ALT](img/example/graphemes/ꝑ/46_0.png)
![ALT](img/example/graphemes/ꝑ/46_1.png)
![ALT](img/example/graphemes/ꝑ/46_2.png)
![ALT](img/example/graphemes/ꝑ/46_3.png)
![ALT](img/example/graphemes/ꝑ/46_4.png)

## Caveats

Les lignes ne sont pas toujours bien liées aux zones, et cela peut entraîner une erreur
que le script ne sait gérer en l'état: dans ce cas, il faut faire l'impasse sur les zones.