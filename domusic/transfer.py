from pydub import AudioSegment
import os

import music21 as ms
notetxt = []
file = ms.converter.parse('output.mid')
print(file)
notes = file.flat.notes
for element in notes:
    # 如果是 Note 类型，取音调
    # 如果是 Chord 类型，取音调的序号,存int类型比较容易处理
    if isinstance(element, ms.note.Note):
        notetxt.append(str(element.pitch))
    elif isinstance(element, ms.chord.Chord):
        notetxt.append('.'.join(str(n) for n in element.normalOrder))

print(notetxt)

from pydub import AudioSegment

wav_path = '../chimeNote/'
musiclist = AudioSegment.empty()
second_silence = AudioSegment.silent(duration=500)
for i in notetxt:
    note_wav = ''
    for j in i:
        if j in ['C', 'D', 'E', 'F', 'G', 'A', 'B'] or j in ['3', '4', '5']:
            note_wav += j
    file_wav = wav_path + note_wav + '.wav'
    sound = AudioSegment.from_file(file_wav, format='wav')
    musiclist += sound
musiclist.export('./music.wav', format='wav')
song = AudioSegment.from_wav("./music.wav")
#对音频进行切片（从已经导入的音频里提取片段）
ten_seconds = 10 * 1000
first_10_seconds = song[:ten_seconds]
last_5_seconds = song[-5000:]
beginning = first_10_seconds
end = last_5_seconds
without_the_middle = beginning + end
without_the_middle.export('../../../music/music_final.wav', format="wav")
