"""CrackVision : detection de fissures dans le beton.

Trois modeles partagent les memes donnees et le meme protocole d'evaluation :
- YOLOX (detection supervisee de boites englobantes) ;
- un autoencodeur convolutif entraine uniquement sur des tuiles saines ;
- un modele hybride qui combine les deux.
"""

__version__ = "2.0.0"
