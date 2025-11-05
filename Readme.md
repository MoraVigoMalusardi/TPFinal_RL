# Clonar repositorio ai-economist
git clone https://github.com/salesforce/ai-economist.git

# Crea un nuevo entorno con Python 3.8 usando conda (que ya tienes instalado)
conda create -n ai-economist python=3.8

# Activa el nuevo entorno
conda activate ai-economist

# Navega al directorio de ai-economist
cd ~/Documents/2025/Reinforcement\ Learning/TPFinal_RL/ai-economist

# Instala ai-economist
pip install -e .

# Si no funciona
pip install setuptools==58.0.4 pip==21.3.1
pip install -e .

# Verificar instalacion
python prueba.py



