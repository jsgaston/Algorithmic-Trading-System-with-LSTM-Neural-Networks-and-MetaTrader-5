#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para iniciar el bot de trading LSTM para MT5
"""

import os
import sys
import logging
import inspect
import subprocess

def check_and_install_requirements():
    """Verifica e instala las dependencias necesarias"""
    try:
        # Obtener el directorio donde se encuentra este script
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        
        # Ruta al archivo requirements.txt
        requirements_path = os.path.join(current_dir, "requirements.txt")
        
        if os.path.exists(requirements_path):
            print("Verificando e instalando dependencias necesarias...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            print("Dependencias instaladas correctamente.")
        else:
            print(f"ADVERTENCIA: No se encontró el archivo requirements.txt en {current_dir}")
    except Exception as e:
        print(f"Error al instalar dependencias: {e}")
        print("Por favor, instale manualmente las dependencias con: pip install -r requirements.txt")

if __name__ == "__main__":
    # Instalar dependencias
    check_and_install_requirements()
    
    # Importar MT5LSTMTrader después de instalar las dependencias
    try:
        from mt5_lstm_trader import MT5LSTMTrader
    except ImportError as e:
        print(f"Error al importar MT5LSTMTrader: {e}")
        print("Asegúrese de que todas las dependencias estén instaladas correctamente.")
        sys.exit(1)
    
    # Obtener el directorio donde se encuentra este script
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # Configurar registro
    log_path = os.path.join(current_dir, "trading_bot.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Iniciando bot de trading LSTM para MT5")
    
    # Construir ruta al archivo config.json dentro del directorio del script
    config_path = os.path.join(current_dir, "config.json")
    
    # Verificar archivos necesarios
    if not os.path.exists(config_path):
        logger.error(f"No se encontró el archivo config.json en {current_dir}")
        print(f"Error: No se encontró el archivo config.json en {current_dir}")
        sys.exit(1)
    
    try:
        # Iniciar el bot con la ruta específica al archivo de configuración
        bot = MT5LSTMTrader(config_path=config_path)
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")
        print("\nBot detenido por el usuario")
    except Exception as e:
        logger.error(f"Error al iniciar el bot: {e}")
        print(f"Error: {e}")
        sys.exit(1)