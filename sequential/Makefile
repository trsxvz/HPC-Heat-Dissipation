# Nom du fichier exécutable
TARGET = codesequentiel

# Compiler
CC = gcc

# Options de compilation
CFLAGS = -lm

# Dépendance
SRC = codeSequentielCorrige.c

# Règle par défaut
all: $(TARGET)

# Règle pour créer l'exécutable
$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(CFLAGS)

# Règle pour nettoyer les fichiers compilés
clean:
	rm -f $(TARGET)

