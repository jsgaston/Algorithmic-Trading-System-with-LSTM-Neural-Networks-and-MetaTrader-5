import json
import logging
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import MetaTrader5 as mt5
import telegram
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5LSTMTrader:
    def __init__(self, config_path='config.json'):
        # Guardar el directorio de trabajo
        self.script_dir = os.path.dirname(os.path.abspath(config_path))
        logger.info(f"Directorio de trabajo: {self.script_dir}")
        
        # Cargar configuración
        self.load_config(config_path)
        
        # Inicializar MT5
        self.init_mt5()
        
        # Inicializar Telegram Bot
        self.init_telegram()
        
        # Inicializar escaladores
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Variables del modelo
        self.model = None
        self.model_path = os.path.join(self.script_dir, 'lstm_model.h5')
        self.last_trained = None
        
        # Métricas y resultados
        self.metrics = {
            'mse': [],
            'mae': [],
            'r2': []
        }

    def load_config(self, config_path):
        """Carga la configuración desde el archivo JSON"""
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                
            # MetaTrader5 credentials
            self.mt5_login = config['mt5_credentials']['login']
            self.mt5_password = config['mt5_credentials']['password']
            self.mt5_server = config['mt5_credentials']['server']
            self.mt5_path = config['mt5_credentials']['path']
            
            # Trading parameters
            self.symbol = config['symbol']
            self.timeframe = config['timeframe']
            self.timeframe_dict = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5, 
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            self.look_back = config['look_back']
            self.retraining_hours = config['retraining_hours']
            
            # Order parameters
            self.lot_size = config['lot_size']
            self.tp_multiplier = config['tp_multiplier']
            self.sl_multiplier = config['sl_multiplier']
            self.trailing_start_pct = config['trailing_start_pct']
            self.trailing_step_pct = config['trailing_step_pct']
            self.risk_per_trade_pct = config['risk_per_trade_pct']
            
            # Telegram parameters
            self.telegram_token = config['telegram_bot_token']
            self.telegram_chat_id = config['telegram_chat_id']
            
            # Model parameters
            self.confidence_threshold = config['confidence_threshold']
            self.price_change_threshold = config['price_change_threshold']
            self.max_data_points = config['max_data_points']
            
            logger.info(f"Configuración cargada correctamente desde {config_path}")
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {e}")
            raise

    def init_mt5(self):
        """Inicializa la conexión con MetaTrader 5"""
        try:
            if not mt5.initialize(path=self.mt5_path):
                logger.error(f"Error al inicializar MT5: {mt5.last_error()}")
                raise Exception(f"MT5 inicialización fallida: {mt5.last_error()}")
            
            # Login a la cuenta
            if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
                logger.error(f"Error al iniciar sesión en MT5: {mt5.last_error()}")
                mt5.shutdown()
                raise Exception(f"MT5 login fallido: {mt5.last_error()}")
            
            logger.info(f"Conectado a MT5 como {self.mt5_login}")
        except Exception as e:
            logger.error(f"Error en la inicialización de MT5: {e}")
            raise

    def init_telegram(self):
        """Inicializa el bot de Telegram"""
        try:
            self.telegram_bot = telegram.Bot(token=self.telegram_token)
            logger.info("Bot de Telegram inicializado")
            self.send_telegram_message("🤖 Bot de Trading iniciado correctamente!")
        except Exception as e:
            logger.error(f"Error al inicializar Telegram: {e}")
            self.telegram_bot = None

    def send_telegram_message(self, message, image_path=None):
        """Envía un mensaje a Telegram, opcionalmente con una imagen"""
        if not self.telegram_bot:
            logger.warning("Bot de Telegram no inicializado, no se enviará el mensaje")
            return
        
        try:
            self.telegram_bot.send_message(chat_id=self.telegram_chat_id, text=message)
            
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as img:
                    self.telegram_bot.send_photo(chat_id=self.telegram_chat_id, photo=img)
            
            logger.info("Mensaje enviado a Telegram correctamente")
        except Exception as e:
            logger.error(f"Error al enviar mensaje a Telegram: {e}")

    def get_historical_data(self, num_bars=None):
        """Obtiene datos históricos de MT5"""
        if num_bars is None:
            num_bars = self.max_data_points
        
        try:
            # Obtener datos del timeframe especificado
            mt5_timeframe = self.timeframe_dict.get(self.timeframe, mt5.TIMEFRAME_H1)
            bars = mt5.copy_rates_from_pos(self.symbol, mt5_timeframe, 0, num_bars)
            
            if bars is None or len(bars) == 0:
                logger.error(f"No se pudieron obtener datos históricos: {mt5.last_error()}")
                raise Exception(f"Error al obtener datos históricos: {mt5.last_error()}")
            
            # Convertir a DataFrame
            df = pd.DataFrame(bars)
            
            # Mostrar las columnas disponibles para propósitos de depuración
            logger.info(f"Columnas disponibles en datos de MT5: {', '.join(df.columns)}")
            
            # Convertir tiempo a formato datetime y establecer como índice
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # En MT5, el volumen está en la columna tick_volume
            if 'tick_volume' in df.columns and 'volume' not in df.columns:
                df['volume'] = df['tick_volume']
                logger.info("Columna 'tick_volume' mapeada a 'volume'")
            
            # Verificar que todas las columnas OHLC estén presentes con sus nombres esperados
            ohlc_mappings = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close'
            }
            
            for orig, renamed in ohlc_mappings.items():
                if orig in df.columns and renamed not in df.columns:
                    df[renamed] = df[orig]
                    logger.info(f"Columna '{orig}' mapeada a '{renamed}'")
            
            # Verificar que todas las columnas requeridas estén presentes
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Si falta 'volume', intentamos crear una columna ficticia (mejor que fallar)
                if 'volume' in missing_columns:
                    logger.warning("Columna 'volume' no encontrada, creando columna ficticia con valores 1")
                    df['volume'] = 1
                    missing_columns.remove('volume')
                
                # Si todavía faltan columnas críticas, lanzamos error
                if missing_columns:
                    raise Exception(f"Columnas requeridas no encontradas: {', '.join(missing_columns)}. Columnas disponibles: {', '.join(df.columns)}")
            
            logger.info(f"Datos históricos obtenidos: {len(df)} barras")
            return df
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            raise

    def prepare_data(self, df):
        """Prepara los datos para el entrenamiento del modelo LSTM"""
        try:
            # Verificar y renombrar columnas si es necesario
            if 'tick_volume' in df.columns and 'volume' not in df.columns:
                df['volume'] = df['tick_volume']
                logger.info("Columna 'tick_volume' renombrada a 'volume'")
            
            # Verificar que todas las columnas necesarias estén presentes
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    available_cols = ', '.join(df.columns)
                    raise Exception(f"La columna '{col}' no está presente en el DataFrame. Columnas disponibles: {available_cols}")
            
            # Seleccionar características y objetivo
            features = df[['open', 'high', 'low', 'close', 'volume']].values
            targets = df[['high', 'low', 'close']].values
            
            # Normalizar datos
            features_scaled = self.scaler_X.fit_transform(features)
            targets_scaled = self.scaler_y.fit_transform(targets)
            
            # Crear secuencias para LSTM
            X, y = [], []
            for i in range(self.look_back, len(features_scaled)):
                X.append(features_scaled[i-self.look_back:i])
                y.append(targets_scaled[i])
            
            X, y = np.array(X), np.array(y)
            
            # Dividir en train y test (80/20)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            logger.info(f"Datos preparados - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error al preparar los datos: {e}")
            raise

    def build_model(self, input_shape):
        """Construye el modelo LSTM"""
        try:
            model = Sequential()
            
            # Primera capa LSTM
            model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            
            # Segunda capa LSTM
            model.add(LSTM(64))
            model.add(Dropout(0.2))
            
            # Capa de salida (predice high, low, close)
            model.add(Dense(3))
            
            # Compilar modelo
            model.compile(optimizer='adam', loss='mse')
            
            logger.info("Modelo LSTM construido")
            return model
        except Exception as e:
            logger.error(f"Error al construir el modelo: {e}")
            raise

    def train_model(self):
        """Entrena o reentrea el modelo LSTM"""
        try:
            # Obtener datos históricos
            df = self.get_historical_data()
            
            # Preparar datos
            X_train, y_train, X_test, y_test = self.prepare_data(df)
            
            # Construir o cargar modelo
            if os.path.exists(self.model_path) and self.model is None:
                try:
                    self.model = load_model(self.model_path)
                    logger.info("Modelo existente cargado correctamente")
                except:
                    logger.warning("No se pudo cargar el modelo existente, creando uno nuevo")
                    self.model = self.build_model(X_train.shape[1:])
            elif self.model is None:
                self.model = self.build_model(X_train.shape[1:])
            
            # Callbacks para entrenamiento
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(self.model_path, save_best_only=True)
            ]
            
            # Entrenar modelo
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluar modelo
            y_pred_scaled = self.model.predict(X_test)
            y_test_unscaled = self.scaler_y.inverse_transform(y_test)
            y_pred_unscaled = self.scaler_y.inverse_transform(y_pred_scaled)
            
            # Calcular métricas
            mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
            mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
            r2 = r2_score(y_test_unscaled.flatten(), y_pred_unscaled.flatten())
            
            # Guardar métricas
            self.metrics['mse'].append(mse)
            self.metrics['mae'].append(mae)
            self.metrics['r2'].append(r2)
            
            # Visualizar resultados
            self._plot_training_history(history)
            self._plot_predictions(y_test_unscaled, y_pred_unscaled)
            
            # Actualizar tiempo de entrenamiento
            self.last_trained = datetime.now()
            
            # Enviar resultados a Telegram
            message = (
                "🔄 Modelo reentrenado\n"
                f"MSE: {mse:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"R²: {r2:.4f}\n"
                f"Cantidad de datos: {len(df)}"
            )
            
            # Usar rutas absolutas para los archivos de imagen
            predictions_img = os.path.join(self.script_dir, 'predictions.png')
            self.send_telegram_message(message, predictions_img)
            
            logger.info(f"Modelo entrenado correctamente - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            return history
        except Exception as e:
            logger.error(f"Error al entrenar el modelo: {e}")
            raise

    def _plot_training_history(self, history):
        """Visualiza la historia de entrenamiento"""
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            # Guardar con ruta absoluta
            history_img = os.path.join(self.script_dir, 'training_history.png')
            plt.savefig(history_img)
            plt.close()
        except Exception as e:
            logger.error(f"Error al visualizar la historia de entrenamiento: {e}")

    def _plot_predictions(self, y_true, y_pred, n_samples=100):
        """Visualiza las predicciones vs valores reales"""
        try:
            # Limitar a las últimas n_samples para mejor visualización
            if len(y_true) > n_samples:
                y_true = y_true[-n_samples:]
                y_pred = y_pred[-n_samples:]
            
            # Crear figura
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            titles = ['High', 'Low', 'Close']
            
            for i, title in enumerate(titles):
                axes[i].plot(y_true[:, i], label=f'Real {title}', color='blue')
                axes[i].plot(y_pred[:, i], label=f'Predicción {title}', color='red', linestyle='--')
                axes[i].set_title(f'Predicciones vs Reales - {title}')
                axes[i].set_xlabel('Tiempo')
                axes[i].set_ylabel('Precio')
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            
            # Guardar con ruta absoluta
            predictions_img = os.path.join(self.script_dir, 'predictions.png')
            plt.savefig(predictions_img)
            plt.close()
        except Exception as e:
            logger.error(f"Error al visualizar las predicciones: {e}")

    def _plot_metrics_over_time(self):
        """Visualiza la evolución de las métricas a lo largo del tiempo"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['mse'], label='MSE')
            plt.plot(self.metrics['mae'], label='MAE')
            plt.plot(self.metrics['r2'], label='R²')
            plt.title('Evolución de Métricas')
            plt.xlabel('Reentrenamientos')
            plt.ylabel('Valor')
            plt.legend()
            
            # Guardar con ruta absoluta
            metrics_img = os.path.join(self.script_dir, 'metrics_evolution.png')
            plt.savefig(metrics_img)
            plt.close()
            
            self.send_telegram_message("📊 Evolución de métricas del modelo", metrics_img)
        except Exception as e:
            logger.error(f"Error al visualizar las métricas: {e}")

    def predict_next_candle(self):
        """Predice el próximo valor de high, low, close"""
        try:
            # Verificar si el modelo existe
            if self.model is None:
                logger.warning("No hay modelo para predecir, entrenando uno nuevo")
                self.train_model()
            
            # Obtener los últimos datos
            df = self.get_historical_data(self.look_back + 1)
            
            # Preparar datos para la predicción
            features = df[['open', 'high', 'low', 'close', 'volume']].values
            features_scaled = self.scaler_X.transform(features)
            
            # Crear secuencia para LSTM
            X_pred = np.array([features_scaled])
            
            # Predecir
            y_pred_scaled = self.model.predict(X_pred)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            
            # Obtener valores actuales
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_close = df['close'].iloc[-1]
            
            # Calcular porcentajes de cambio
            pred_high, pred_low, pred_close = y_pred[0]
            
            high_change_pct = ((pred_high - current_close) / current_close) * 100
            low_change_pct = ((pred_low - current_close) / current_close) * 100
            close_change_pct = ((pred_close - current_close) / current_close) * 100
            
            logger.info(f"Predicción - High: {pred_high:.5f} ({high_change_pct:.2f}%), "
                        f"Low: {pred_low:.5f} ({low_change_pct:.2f}%), "
                        f"Close: {pred_close:.5f} ({close_change_pct:.2f}%)")
            
            return {
                'pred_high': pred_high,
                'pred_low': pred_low,
                'pred_close': pred_close,
                'high_change_pct': high_change_pct,
                'low_change_pct': low_change_pct,
                'close_change_pct': close_change_pct,
                'current_close': current_close
            }
        except Exception as e:
            logger.error(f"Error al predecir el próximo valor: {e}")
            return None

    def place_order(self, prediction):
        """Coloca una orden basada en la predicción"""
        try:
            # Extraer datos de la predicción
            close_change_pct = prediction['close_change_pct']
            current_close = prediction['current_close']
            
            # Determinar dirección basada en el porcentaje de cambio del precio de cierre
            if abs(close_change_pct) < self.price_change_threshold:
                logger.info(f"No se coloca orden - Cambio de precio ({close_change_pct:.2f}%) por debajo del umbral ({self.price_change_threshold}%)")
                return None
            
            # Determinar tipo de orden
            order_type = mt5.ORDER_TYPE_BUY if close_change_pct > 0 else mt5.ORDER_TYPE_SELL
            direction = "COMPRA" if order_type == mt5.ORDER_TYPE_BUY else "VENTA"
            
            # Calcular stop loss y take profit
            price_info = mt5.symbol_info_tick(self.symbol)
            
            if price_info is None:
                logger.error(f"No se pudo obtener información de precio para {self.symbol}")
                return None
            
            current_price = price_info.ask if order_type == mt5.ORDER_TYPE_BUY else price_info.bid
            
            # Calcular pip value
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"No se pudo obtener información del símbolo {self.symbol}")
                return None
            
            pip_value = 10**(-symbol_info.digits)
            
            # Calcular SL y TP en pips
            atr = self._calculate_atr(20)  # ATR de 20 periodos
            sl_pips = atr * self.sl_multiplier
            tp_pips = atr * self.tp_multiplier
            
            # Convertir pips a precio
            if order_type == mt5.ORDER_TYPE_BUY:
                sl_price = current_price - sl_pips
                tp_price = current_price + tp_pips
            else:
                sl_price = current_price + sl_pips
                tp_price = current_price - tp_pips
            
            # Preparar la solicitud de orden
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": order_type,
                "price": current_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": 12345,
                "comment": f"LSTM Prediction: {close_change_pct:.2f}%",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar la orden
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Error al colocar orden: {result.comment}")
                self.send_telegram_message(f"❌ Error al colocar orden: {result.comment}")
                return None
            
            # Registrar y notificar sobre la orden
            logger.info(f"Orden colocada: {direction} {self.lot_size} {self.symbol} @ {current_price}, SL: {sl_price}, TP: {tp_price}")
            
            # Enviar mensaje a Telegram
            message = (
                f"🔔 NUEVA ORDEN: {direction}\n"
                f"Símbolo: {self.symbol}\n"
                f"Lote: {self.lot_size}\n"
                f"Precio de entrada: {current_price}\n"
                f"Stop Loss: {sl_price}\n"
                f"Take Profit: {tp_price}\n"
                f"Predicción de cambio: {close_change_pct:.2f}%\n"
                f"Predicción High: {prediction['pred_high']:.5f} ({prediction['high_change_pct']:.2f}%)\n"
                f"Predicción Low: {prediction['pred_low']:.5f} ({prediction['low_change_pct']:.2f}%)\n"
                f"Predicción Close: {prediction['pred_close']:.5f} ({prediction['close_change_pct']:.2f}%)"
            )
            self.send_telegram_message(message)
            
            return result
        except Exception as e:
            logger.error(f"Error al colocar la orden: {e}")
            self.send_telegram_message(f"❌ Error al colocar orden: {str(e)}")
            return None

    def _calculate_atr(self, period=14):
        """Calcula el ATR (Average True Range) para el periodo especificado"""
        try:
            # Obtener datos históricos
            df = self.get_historical_data(period + 1)
            
            # Calcular True Range
            df['high-low'] = df['high'] - df['low']
            df['high-prev_close'] = abs(df['high'] - df['close'].shift(1))
            df['low-prev_close'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
            
            # Calcular ATR
            atr = df['tr'].mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error al calcular ATR: {e}")
            return 0.001  # Valor por defecto pequeño

    def run(self):
        """Ejecuta el bot de trading"""
        try:
            logger.info("Iniciando bot de trading...")
            self.send_telegram_message("🚀 Bot de Trading iniciado!")
            
            # Entrenar el modelo inicial
            self.train_model()
            
            while True:
                try:
                    # Verificar si es necesario reentrenar el modelo
                    if (self.last_trained is None or 
                        datetime.now() - self.last_trained > timedelta(hours=self.retraining_hours)):
                        logger.info(f"Reentrenando modelo (último entrenamiento: {self.last_trained})")
                        self.train_model()
                        
                        # Graficar evolución de métricas si hay suficientes datos
                        if len(self.metrics['mse']) > 1:
                            self._plot_metrics_over_time()
                    
                    # Realizar predicción
                    prediction = self.predict_next_candle()
                    
                    if prediction:
                        # Enviar mensaje de predicción a Telegram
                        message = (
                            f"🔮 PREDICCIÓN:\n"
                            f"Símbolo: {self.symbol}\n"
                            f"Timeframe: {self.timeframe}\n"
                            f"Predicción High: {prediction['pred_high']:.5f} ({prediction['high_change_pct']:.2f}%)\n"
                            f"Predicción Low: {prediction['pred_low']:.5f} ({prediction['low_change_pct']:.2f}%)\n"
                            f"Predicción Close: {prediction['pred_close']:.5f} ({prediction['close_change_pct']:.2f}%)"
                        )
                        self.send_telegram_message(message)
                        
                        # Colocar orden si el cambio es significativo
                        if abs(prediction['close_change_pct']) >= self.price_change_threshold:
                            self.place_order(prediction)
                    
                    # Esperar hasta el próximo periodo
                    wait_time = self._get_wait_time()
                    logger.info(f"Esperando {wait_time} segundos hasta la próxima vela...")
                    time.sleep(wait_time)
                
                except Exception as e:
                    logger.error(f"Error durante la ejecución: {e}")
                    self.send_telegram_message(f"⚠️ Error durante la ejecución: {str(e)}")
                    time.sleep(60)  # Esperar un minuto antes de reintentar
                    
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
            self.send_telegram_message("🛑 Bot detenido por el usuario")
        except Exception as e:
            logger.error(f"Error fatal: {e}")
            self.send_telegram_message(f"🚨 ERROR FATAL: {str(e)}")
        finally:
            # Limpiar recursos
            if mt5.initialize():  # Check if MT5 is initialized
                mt5.shutdown()
                logger.info("MT5 desconectado")
            else:
                logger.info("MT5 ya estaba desconectado")

if __name__ == "__main__":
    try:
        # Crear y ejecutar el bot
        bot = MT5LSTMTrader()
        bot.run()
    except Exception as e:
        logging.error(f"Error al iniciar el bot: {e}")