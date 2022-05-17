import random
import time

# Einleitung
print("***************************"); time.sleep (0.5)
print("* Schere | Stein | Papier *"); time.sleep (0.5)
print("***************************/n/n"); time.sleep (0.5)

# Variablen
figueren = ("Schere" , "Stein" , "Papier")
spielen = true

while spielen:

	# Spielerfigur auswählen
	spielerauswahl = 0
	while spielerauswahl not in (1,2,3):
		spielerauswahl = int(input("[1]Schere [2]Stein [3]Papier⁄n"))
	Spielerfigur = figueren[spielerauswahl - 1]

	#Computerfigur auswählen
	Computerfigur = figuren[random.randint(0,2)]

	#Siger ermitteln
	if Spielerfigur == Computerfigur:
			print("Unentschieden! Computer wählte", Computerfigur)
		else:

			if Spielerfigur == "Schere":
				if Computerfigur == "Stein":
					print("Verloren!  Computer wählte", Computerfigur)
				else:
					print ("Gewonnen Computer wählte", Computerfigur)	
					
				if Spielerfigur == "Stein":
					if Computerfigur == "Schere":
						print ("Gewonnen! Computer wählte", Computerfigur)
					else:
						print ("Verloren! Computer wählte", Computerfigur)

				if Spielerfigur == "Papier":
					if Computerfigur == "Schere":
						print("Verloren! Computer wählte", Computerfigur)
					else:
						print(" Gewonnen! Computer wählte, Computerfigur)

		#Restart
		time.sleep(1)
		entscheidung = ""
		while  entscheidung not in ("y", "n"):
			entscheidung = input("/n Nochmal Spielen? [y]Ja  [n]Nein")

		if(entscheidung == "n"):
			spielen = False








