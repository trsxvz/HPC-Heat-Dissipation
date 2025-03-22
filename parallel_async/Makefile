# Nom du fichier exécutable
TARGET = parallelisation1Dasynchrone

# Compiler
CC = mpicc

# Options de compilation
CFLAGS = -lm

# Dépendance
SRC = parallelisation1Dasynchrone.c

# Règle par défaut
all: $(TARGET)

# Règle pour créer l'exécutable
$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(CFLAGS)

# Règle pour nettoyer les fichiers compilés
clean:
	rm -f $(TARGET)

