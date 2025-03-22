# Nom du fichier exécutable
TARGET = parallelisation1D

# Compiler
CC = mpicc

# Options de compilation
CFLAGS = -lm

# Dépendance
SRC = parallelisation1D.c

# Règle par défaut
all: $(TARGET)

# Règle pour créer l'exécutable
$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(CFLAGS)

# Règle pour nettoyer les fichiers compilés
clean:
	rm -f $(TARGET)

