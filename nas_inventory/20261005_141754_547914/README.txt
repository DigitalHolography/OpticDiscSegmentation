patients.csv : codes reconnus, triés par nombre de nouvelles acquisitions décroissant.
n_images compte les fichiers M0 ; n_acquisitions compte leurs dossiers parents distincts.
new_images.csv relie chaque image au patient et à l'acquisition.
identity_review.csv contient les identités à vérifier ; elles ne comptent pas dans patients.csv.
present_in_reference = code déjà reconnu dans la liste ; not_identified_in_reference =
code non reconnu dans la liste, pas une preuve qu'il s'agit d'un nouveau patient réel.
Les exclusions portent sur les chemins d'acquisition, pas sur le contenu des images.
Les variantes ambiguës peuvent cacher des patients communs. Aucun label n'est généré.
Les erreurs et liens ignorés sont listés ; le scan est alors incomplet.
new_paths.txt contient tous les candidats, y compris les identités à vérifier.
