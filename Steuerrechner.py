#Steuer Rechner

#Eingabe
print("Wie viel ist ihr Brutto gehalt monatlich")
x = input()
zahl = float(x)
a = float(0.19)
z = zahl * a

# Bruttolohn werden mit den Steuern von 19% gerechnet

print("Ihr Lohn:", x, "*" ,a )
print("Ihre Monatliche Steuerausgabe liegt bei", z)
