Ce script permet de récupérer l'image de chaque graphème d'un manuscrit transcrit *automatiquement* avec eScriptorium (a venir: avec Kraken).

Il utilise l'API d'eScriptorium pour récupérer les informations, les images, et extraires les graphèmes à des fins d'étude et/ou de comparaison.

La proportion de graphèmes conservés permet de prélever des échantillons.

Il est nécessaire de disposer de modèles de transcription corrects pour récupérer des images exploitables et fiables. 
Les graphèmes sont classées en reprenant les classes reconnues par le modèle: le script reproduira donc
les biais du modèle éventuels.

Un ajustement de la hauteur et de la largeur des images extraites est probablement nécessaire (à venir: 
distinction des lettres en configuration par
leur hauteur par rapport à la ligne)

Si une zone est préférée, elle sera indiquée dans le fichier configuration. Si le document
n'est pas zoné, marquer ``null``.

## Utilisation

`python3 retrieve_graphemes.py fichier.conf`

Il est nécessaire de lancer une première fois le script à blanc pour récupérer
les identifiants de transcription et de zone.