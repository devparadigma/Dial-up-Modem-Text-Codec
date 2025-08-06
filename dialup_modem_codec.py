import numpy as np
import wave
import struct
import os
import sys
from scipy.io.wavfile import write, read
from scipy.signal import hilbert, find_peaks

class DialupModem:
  def __init__(self, sample_rate=8000, baud_rate=300):
      """
      Инициализация параметров модема
      sample_rate: частота дискретизации (Гц)
      baud_rate: скорость передачи (бод)
      """
      self.sample_rate = sample_rate
      self.baud_rate = baud_rate
      self.bit_duration = sample_rate / baud_rate  # количество сэмплов на бит
      
      # Частоты для FSK модуляции (стандарт Bell 103)
      self.freq_0 = 1070  # частота для бита '0'
      self.freq_1 = 1270  # частота для бита '1'
      
      self.debug_mode = False  # Режим отладки
      
  def text_to_bits(self, text):
      """Преобразование текста в биты с поддержкой UTF-8"""
      bits = []
      
      # Преобразуем текст в байты UTF-8
      try:
          text_bytes = text.encode('utf-8')
      except UnicodeEncodeError:
          print("❌ Ошибка кодирования текста в UTF-8")
          return []
      
      if self.debug_mode:
          print(f"🔍 Текст в байтах UTF-8: {list(text_bytes)}")
      
      # Добавляем маркер начала (4 байта)
      start_marker = b'\xAA\x55\xAA\x55'  # Легко различимый паттерн
      
      # Добавляем длину данных (2 байта, little-endian)
      data_length = len(text_bytes)
      length_bytes = data_length.to_bytes(2, 'little')
      
      # Формируем полный пакет данных
      full_data = start_marker + length_bytes + text_bytes
      
      if self.debug_mode:
          print(f"🔍 Полный пакет: {list(full_data)}")
          print(f"🔍 Длина данных: {data_length} байт")
      
      # Преобразуем каждый байт в биты (старший бит первый)
      for byte_val in full_data:
          for i in range(7, -1, -1):  # От старшего к младшему биту
              bits.append((byte_val >> i) & 1)
      
      return bits
  
  def bits_to_text(self, bits):
      """Преобразование битов в текст с поддержкой UTF-8"""
      if len(bits) < 48:  # Минимум для маркера + длины
          if self.debug_mode:
              print("❌ Недостаточно битов для декодирования")
          return ""
      
      # Преобразуем биты в байты
      bytes_data = []
      for i in range(0, len(bits) - 7, 8):
          if i + 8 <= len(bits):
              byte_val = 0
              for j in range(8):
                  byte_val = (byte_val << 1) | bits[i + j]
              bytes_data.append(byte_val)
      
      if self.debug_mode:
          print(f"🔍 Декодированные байты: {bytes_data[:20]}..." if len(bytes_data) > 20 else f"🔍 Декодированные байты: {bytes_data}")
      
      # Ищем маркер начала
      start_marker = [0xAA, 0x55, 0xAA, 0x55]
      start_pos = -1
      
      for i in range(len(bytes_data) - 3):
          if bytes_data[i:i+4] == start_marker:
              start_pos = i + 4
              break
      
      if start_pos == -1:
          if self.debug_mode:
              print("❌ Маркер начала не найден")
          return ""
      
      if start_pos + 2 >= len(bytes_data):
          if self.debug_mode:
              print("❌ Недостаточно данных после маркера")
          return ""
      
      # Читаем длину данных
      data_length = bytes_data[start_pos] + (bytes_data[start_pos + 1] << 8)
      data_start = start_pos + 2
      
      if self.debug_mode:
          print(f"🔍 Найден маркер на позиции {start_pos - 4}")
          print(f"🔍 Длина данных: {data_length} байт")
      
      if data_start + data_length > len(bytes_data):
          if self.debug_mode:
              print(f"❌ Недостаточно данных: нужно {data_length}, доступно {len(bytes_data) - data_start}")
          # Берем сколько есть
          data_length = len(bytes_data) - data_start
      
      # Извлекаем данные
      text_bytes = bytes(bytes_data[data_start:data_start + data_length])
      
      if self.debug_mode:
          print(f"🔍 Извлеченные байты: {list(text_bytes)}")
      
      # Преобразуем байты в текст UTF-8
      try:
          decoded_text = text_bytes.decode('utf-8', errors='ignore')
          return decoded_text
      except UnicodeDecodeError as e:
          if self.debug_mode:
              print(f"❌ Ошибка декодирования UTF-8: {e}")
          return ""
  
  def encode_to_audio(self, text, output_file):
      """Кодирование текста в звуковой файл"""
      print(f"\n📡 Кодирование текста: '{text}'")
      
      # Преобразуем текст в биты
      bits = self.text_to_bits(text)
      if not bits:
          print("❌ Ошибка преобразования текста в биты")
          return False
          
      print(f"📊 Количество битов: {len(bits)}")
      
      if self.debug_mode:
          print(f"🔍 Первые биты: {bits[:32]}..." if len(bits) > 32 else f"🔍 Все биты: {bits}")
      
      # Создаем звуковой сигнал
      audio_data = []
      
      # Добавляем небольшую тишину в начале
      silence_duration = int(0.1 * self.sample_rate)
      audio_data.extend(np.zeros(silence_duration))
      
      # Кодируем каждый бит
      for i, bit in enumerate(bits):
          freq = self.freq_1 if bit == 1 else self.freq_0
          
          # Создаем синусоиду для данного бита
          t = np.linspace(0, 1/self.baud_rate, int(self.bit_duration))
          bit_signal = 0.8 * np.sin(2 * np.pi * freq * t)  # Амплитуда 0.8
          
          audio_data.extend(bit_signal)
          
          # Показываем прогресс
          if (i + 1) % 50 == 0 or i == len(bits) - 1:
              progress = (i + 1) / len(bits) * 100
              print(f"⏳ Прогресс кодирования: {progress:.1f}%")
      
      # Добавляем тишину в конце
      audio_data.extend(np.zeros(silence_duration))
      
      # Преобразуем в numpy array
      audio_data = np.array(audio_data)
      
      # Сохраняем в WAV файл
      write(output_file, self.sample_rate, (audio_data * 32767).astype(np.int16))
      print(f"✅ Аудио файл сохранен: {output_file}")
      print(f"🕒 Длительность: {len(audio_data) / self.sample_rate:.2f} секунд")
      return True
  
  def correlate_with_frequency(self, signal, freq, sample_rate):
      """Корреляция сигнала с эталонной частотой"""
      t = np.linspace(0, len(signal) / sample_rate, len(signal))
      reference_sin = np.sin(2 * np.pi * freq * t)
      reference_cos = np.cos(2 * np.pi * freq * t)
      
      # Вычисляем корреляцию
      corr_sin = np.sum(signal * reference_sin)
      corr_cos = np.sum(signal * reference_cos)
      
      # Возвращаем мощность (квадрат амплитуды)
      return corr_sin**2 + corr_cos**2
  
  def decode_from_audio(self, input_file):
      """Декодирование звукового файла в текст"""
      print(f"\n🎵 Декодирование файла: {input_file}")
      
      try:
          # Проверяем существование файла
          if not os.path.exists(input_file):
              print(f"❌ Ошибка: файл '{input_file}' не найден!")
              return ""
          
          # Читаем WAV файл
          sample_rate, audio_data = read(input_file)
          
          # Нормализуем данные
          if audio_data.dtype == np.int16:
              audio_data = audio_data.astype(np.float32) / 32767.0
          elif audio_data.dtype == np.int32:
              audio_data = audio_data.astype(np.float32) / 2147483647.0
          else:
              audio_data = audio_data.astype(np.float32)
          
          print(f"📊 Длительность файла: {len(audio_data) / sample_rate:.2f} секунд")
          print(f"📊 Частота дискретизации: {sample_rate} Гц")
          
          # Адаптируем параметры под частоту файла
          actual_bit_duration = sample_rate / self.baud_rate
          samples_per_bit = int(actual_bit_duration)
          
          print(f"🔍 Сэмплов на бит: {samples_per_bit}")
          
          # Убираем тишину в начале
          threshold = 0.01
          start_idx = 0
          for i in range(len(audio_data)):
              if abs(audio_data[i]) > threshold:
                  start_idx = max(0, i - samples_per_bit//4)  # Небольшой отступ назад
                  break
          
          audio_data = audio_data[start_idx:]
          print(f"🔍 Начало сигнала на позиции: {start_idx / sample_rate:.3f} сек")
          
          # Декодируем биты
          bits = []
          total_samples = len(audio_data)
          
          print(f"🔍 Начинаем декодирование битов...")
          
          for i in range(0, total_samples - samples_per_bit, samples_per_bit):
              # Извлекаем сегмент для одного бита
              segment = audio_data[i:i + samples_per_bit]
              
              if len(segment) < samples_per_bit:
                  break
              
              # Используем корреляционный метод для определения частоты
              power_0 = self.correlate_with_frequency(segment, self.freq_0, sample_rate)
              power_1 = self.correlate_with_frequency(segment, self.freq_1, sample_rate)
              
              # Определяем бит по большей мощности
              if power_1 > power_0:
                  bits.append(1)
                  if self.debug_mode and len(bits) <= 20:
                      print(f"Бит {len(bits)}: 1 (P0={power_0:.2f}, P1={power_1:.2f})")
              else:
                  bits.append(0)
                  if self.debug_mode and len(bits) <= 20:
                      print(f"Бит {len(bits)}: 0 (P0={power_0:.2f}, P1={power_1:.2f})")
              
              # Показываем прогресс
              if len(bits) % 50 == 0:
                  progress = i / total_samples * 100
                  print(f"⏳ Прогресс декодирования: {min(progress, 100):.1f}%")
          
          print(f"📊 Декодировано битов: {len(bits)}")
          
          if self.debug_mode and len(bits) > 0:
              print(f"🔍 Первые биты: {bits[:32]}..." if len(bits) > 32 else f"🔍 Все биты: {bits}")
          
          # Преобразуем биты в текст
          decoded_text = self.bits_to_text(bits)
          
          if decoded_text:
              print(f"✅ Декодированный текст: '{decoded_text}'")
          else:
              print("❌ Не удалось декодировать текст.")
              if not self.debug_mode:
                  print("💡 Попробуйте включить режим отладки в настройках")
          
          return decoded_text
          
      except Exception as e:
          print(f"❌ Ошибка при декодировании: {e}")
          if self.debug_mode:
              import traceback
              traceback.print_exc()
          return ""

def print_header():
  """Выводит заголовок программы"""
  print("=" * 60)
  print("🖥️  DIAL-UP MODEM TEXT CODEC")
  print("📡 Кодирование/декодирование текста в звуки модема")
  print("=" * 60)

def print_menu():
  """Выводит главное меню"""
  print("\n📋 ГЛАВНОЕ МЕНЮ:")
  print("1️⃣  Закодировать текст в звуковой файл")
  print("2️⃣  Раскодировать звуковой файл в текст")
  print("3️⃣  Настройки модема")
  print("4️⃣  Информация о программе")
  print("5️⃣  Тестирование кодека")
  print("0️⃣  Выход")
  print("-" * 40)

def encode_menu(modem):
  """Меню кодирования"""
  print("\n🔤 КОДИРОВАНИЕ ТЕКСТА")
  print("-" * 30)
  
  # Ввод текста
  text = input("📝 Введите текст для кодирования: ").strip()
  if not text:
      print("❌ Текст не может быть пустым!")
      return
  
  # Ввод имени файла
  default_filename = "encoded_message.wav"
  filename = input(f"💾 Имя выходного файла (по умолчанию: {default_filename}): ").strip()
  if not filename:
      filename = default_filename
  
  # Добавляем расширение .wav если его нет
  if not filename.lower().endswith('.wav'):
      filename += '.wav'
  
  print(f"\n🚀 Начинаем кодирование...")
  success = modem.encode_to_audio(text, filename)
  
  if success:
      print(f"\n🎉 Кодирование завершено успешно!")
      print(f"📁 Файл сохранен как: {filename}")
  
  input("\n👆 Нажмите Enter для продолжения...")

def decode_menu(modem):
  """Меню декодирования"""
  print("\n🎵 ДЕКОДИРОВАНИЕ ЗВУКОВОГО ФАЙЛА")
  print("-" * 35)
  
  # Показываем доступные .wav файлы
  wav_files = [f for f in os.listdir('.') if f.lower().endswith('.wav')]
  if wav_files:
      print("\n📁 Найденные WAV файлы в текущей папке:")
      for i, file in enumerate(wav_files, 1):
          print(f"   {i}. {file}")
      print("\n💡 Можно ввести номер файла или полное имя")
  
  # Ввод имени файла
  filename = input("\n🎧 Введите имя файла или номер для декодирования: ").strip()
  if not filename:
      print("❌ Имя файла не может быть пустым!")
      return
  
  # Проверяем, не ввел ли пользователь номер файла
  try:
      file_num = int(filename)
      if 1 <= file_num <= len(wav_files):
          filename = wav_files[file_num - 1]
          print(f"📁 Выбран файл: {filename}")
      else:
          print("❌ Неверный номер файла!")
          return
  except ValueError:
      # Пользователь ввел имя файла
      if not filename.lower().endswith('.wav'):
          filename += '.wav'
  
  print(f"\n🚀 Начинаем декодирование...")
  decoded_text = modem.decode_from_audio(filename)
  
  if decoded_text:
      print(f"\n🎉 Декодирование завершено успешно!")
      
      # Предлагаем сохранить в файл
      save_choice = input("\n💾 Сохранить результат в текстовый файл? (y/n): ").strip().lower()
      if save_choice in ['y', 'yes', 'да', 'д']:
          text_filename = filename.replace('.wav', '_decoded.txt')
          try:
              with open(text_filename, 'w', encoding='utf-8') as f:
                  f.write(decoded_text)
              print(f"✅ Текст сохранен в файл: {text_filename}")
          except Exception as e:
              print(f"❌ Ошибка сохранения: {e}")
  else:
      print("\n❌ Декодирование не удалось!")
  
  input("\n👆 Нажмите Enter для продолжения...")

def test_codec(modem):
  """Тестирование кодека"""
  print("\n🧪 ТЕСТИРОВАНИЕ КОДЕКА")
  print("-" * 25)
  
  test_texts = [
      "Hello World!",
      "Привет мир!",
      "123456789",
      "Test message",
      "Тест сообщение",
      "English + Русский = OK!"
  ]
  
  print("📋 Доступные тестовые сообщения:")
  for i, text in enumerate(test_texts, 1):
      print(f"   {i}. '{text}'")
  print("   0. Ввести свой текст")
  
  choice = input("\n👉 Выберите тест (0-6): ").strip()
  
  if choice == '0':
      test_text = input("📝 Введите текст для тестирования: ").strip()
      if not test_text:
          print("❌ Текст не может быть пустым!")
          return
  else:
      try:
          idx = int(choice) - 1
          if 0 <= idx < len(test_texts):
              test_text = test_texts[idx]
          else:
              print("❌ Неверный выбор!")
              return
      except ValueError:
          print("❌ Введите корректный номер!")
          return
  
  print(f"\n🧪 Тестируем текст: '{test_text}'")
  
  # Включаем режим отладки на время теста
  old_debug = modem.debug_mode
  modem.debug_mode = True
  
  # Кодирование
  test_filename = "test_codec.wav"
  print(f"\n📡 Этап 1: Кодирование в {test_filename}")
  success = modem.encode_to_audio(test_text, test_filename)
  
  if not success:
      print("❌ Ошибка кодирования!")
      modem.debug_mode = old_debug
      return
  
  # Декодирование
  print(f"\n🎵 Этап 2: Декодирование из {test_filename}")
  decoded_text = modem.decode_from_audio(test_filename)
  
  # Восстанавливаем режим отладки
  modem.debug_mode = old_debug
  
  # Сравнение результатов
  print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТА:")
  print(f"   Исходный текст:    '{test_text}'")
  print(f"   Декодированный:    '{decoded_text}'")
  
  if test_text == decoded_text:
      print("✅ ТЕСТ ПРОЙДЕН! Текст декодирован корректно.")
  else:
      print("❌ ТЕСТ НЕ ПРОЙДЕН! Есть ошибки в декодировании.")
      
      # Анализ ошибок
      if len(test_text) != len(decoded_text):
          print(f"   Длина: исходная={len(test_text)}, декодированная={len(decoded_text)}")
      
      # Показываем различия по символам
      max_len = max(len(test_text), len(decoded_text))
      errors = 0
      for i in range(max_len):
          orig_char = test_text[i] if i < len(test_text) else '∅'
          dec_char = decoded_text[i] if i < len(decoded_text) else '∅'
          if orig_char != dec_char:
              errors += 1
              if errors <= 5:  # Показываем только первые 5 ошибок
                  print(f"   Позиция {i}: '{orig_char}' → '{dec_char}'")
      
      if errors > 5:
          print(f"   ... и еще {errors - 5} ошибок")
  
  input("\n👆 Нажмите Enter для продолжения...")

def settings_menu(modem):
  """Меню настроек"""
  print("\n⚙️  НАСТРОЙКИ МОДЕМА")
  print("-" * 25)
  print(f"📊 Текущие настройки:")
  print(f"   Частота дискретизации: {modem.sample_rate} Гц")
  print(f"   Скорость передачи: {modem.baud_rate} бод")
  print(f"   Частота для '0': {modem.freq_0} Гц")
  print(f"   Частота для '1': {modem.freq_1} Гц")
  print(f"   Режим отладки: {'Включен' if modem.debug_mode else 'Выключен'}")
  
  print(f"\n📋 Варианты настроек:")
  print(f"1. Изменить частоту дискретизации")
  print(f"2. Изменить скорость передачи")
  print(f"3. Переключить режим отладки")
  print(f"4. Сбросить к значениям по умолчанию")
  print(f"0. Назад")
  
  choice = input("\n👉 Выберите опцию: ").strip()
  
  if choice == '1':
      try:
          new_rate = int(input("🎚️  Новая частота дискретизации (рекомендуется 8000): "))
          if new_rate > 0:
              modem.sample_rate = new_rate
              modem.bit_duration = new_rate / modem.baud_rate
              print(f"✅ Частота дискретизации изменена на {new_rate} Гц")
          else:
              print("❌ Частота должна быть положительной!")
      except ValueError:
          print("❌ Введите корректное число!")
  
  elif choice == '2':
      try:
          new_baud = int(input("📡 Новая скорость передачи (рекомендуется 300): "))
          if new_baud > 0:
              modem.baud_rate = new_baud
              modem.bit_duration = modem.sample_rate / new_baud
              print(f"✅ Скорость передачи изменена на {new_baud} бод")
          else:
              print("❌ Скорость должна быть положительной!")
      except ValueError:
          print("❌ Введите корректное число!")
  
  elif choice == '3':
      modem.debug_mode = not modem.debug_mode
      status = "включен" if modem.debug_mode else "выключен"
      print(f"✅ Режим отладки {status}")
  
  elif choice == '4':
      modem.sample_rate = 8000
      modem.baud_rate = 300
      modem.bit_duration = modem.sample_rate / modem.baud_rate
      modem.debug_mode = False
      print("✅ Настройки сброшены к значениям по умолчанию")
  
  if choice in ['1', '2', '3', '4']:
      input("\n👆 Нажмите Enter для продолжения...")

def info_menu():
  """Информация о программе"""
  print("\n📚 ИНФОРМАЦИЯ О ПРОГРАММЕ")
  print("-" * 30)
  print("🖥️  Dial-up Modem Text Codec v4.0")
  print("📡 Программа для кодирования текста в звуки модема")
  print()
  print("🔧 Технические характеристики:")
  print("   • FSK (Frequency Shift Keying) модуляция")
  print("   • Стандарт Bell 103:")
  print("     - Частота для '0': 1070 Гц")
  print("     - Частота для '1': 1270 Гц")
  print("   • Скорость по умолчанию: 300 бод")
  print("   • Формат файлов: WAV (16-бит)")
  print("   • Кодировка: UTF-8")
  print("   • Протокол с маркерами и длиной")
  print()
  print("📋 Возможности:")
  print("   ✅ Кодирование текста в звуковой файл")
  print("   ✅ Декодирование звука обратно в текст")
  print("   ✅ Полная поддержка UTF-8 (русский, эмодзи)")
  print("   ✅ Настраиваемые параметры")
  print("   ✅ Режим отладки")
  print("   ✅ Встроенное тестирование")
  print("   ✅ Надежный протокол передачи")
  print()
  print("🎯 Применение:")
  print("   • Эмуляция работы dial-up модемов")
  print("   • Обучение принципам цифровой связи")
  print("   • Передача данных через аудиоканал")
  
  input("\n👆 Нажмите Enter для продолжения...")

def main():
  """Главная функция программы"""
  # Проверяем наличие необходимых библиотек
  try:
      import numpy as np
      from scipy.io.wavfile import write, read
  except ImportError as e:
      print("❌ Ошибка: не установлены необходимые библиотеки!")
      print("📦 Установите их командой: pip install numpy scipy")
      print(f"Детали ошибки: {e}")
      return
  
  # Создаем экземпляр модема
  modem = DialupModem()
  
  while True:
      print_header()
      print_menu()
      
      choice = input("👉 Выберите пункт меню (0-5): ").strip()
      
      if choice == '1':
          encode_menu(modem)
      elif choice == '2':
          decode_menu(modem)
      elif choice == '3':
          settings_menu(modem)
      elif choice == '4':
          info_menu()
      elif choice == '5':
          test_codec(modem)
      elif choice == '0':
          print("\n👋 До свидания!")
          print("📡 Спасибо за использование Dial-up Modem Codec!")
          break
      else:
          print("❌ Неверный выбор! Попробуйте еще раз.")
          input("👆 Нажмите Enter для продолжения...")
if __name__ == "__main__":
  main()
