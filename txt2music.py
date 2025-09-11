from music21 import *
from pydub import AudioSegment
from midi2audio import FluidSynth
import os


def split_notetxt(notetxt):
    index = 0
    for i in range(len(notetxt)):
        if notetxt[i] != '-':
            index += 1
        else:
            break
    note_type = notetxt[:index]
    note_desc = notetxt[index + 1:]
    return note_type, note_desc


def countDots(note_desc):
    count = 0
    for i in note_desc:
        if i == '.':
            count += 1
    return count


def note2music(notestxt, filename):
    stream1 = stream.Stream()
    m = stream.Measure()
    for i in notestxt:
        note_type, note_desc = split_notetxt(i)
        if note_type == 'clef':
            # TrebleClef为高音谱号，BassClef为低音谱号，AltoClef为中音谱号
            if note_desc[0] == 'G':
                stream1.clef = clef.TrebleClef()
            elif note_desc[0] == 'C':
                stream1.clef = clef.AltoClef()
            elif note_desc[0] == 'F':
                stream1.clef = clef.BassClef()
                if note_type == 'keySignature':
                    stream1.keySignature = key.Key(note_desc)
                if note_type == 'timeSignature':
                    stream1.timeSignature = meter.TimeSignature(note_desc)
        if note_type == 'barline':
            stream1.append(m)
            m = stream.Measure()
        if note_type == 'rest':
            count = countDots(note_desc)
            if 'double_whole' in note_desc:
                note_desc = 'breve'
            if 'whole' in note_desc:
                note_desc = 'whole'
            if 'half' in note_desc:
                note_desc = 'half'
            if 'quarter' in note_desc:
                note_desc = 'quarter'
            if 'eighth' in note_desc:
                note_desc = 'eighth'
            if 'sixteenth' in note_desc:
                note_desc = '16th'
            if 'thirty_second' in note_desc:
                note_desc = '32rd'
            if count == 0:
                m.append(note.Rest(type=note_desc))
            elif count == 1:
                m.append(note.Rest(type=note_desc[:-1], dots=1))
            elif count == 2:
                m.append(note.Rest(type=note_desc[:-2], dots=2))
        if note_type == 'multirest':
            r = note.Rest(type='whole')
            m.repeatAppend(r, int(note_desc))
        if note_type == 'note':
            idx = 0
            count = countDots(note_desc)
            for j in note_desc:
                idx += 1
                if j == '_':
                    note_name = note_desc[:idx - 1]
                    note_type_sp = note_desc[idx:]
                    if 'double_whole' in note_type_sp:
                        note_type_sp = 'breve'
                    if 'whole' in note_type_sp:
                        note_type_sp = 'whole'
                    if 'half' in note_type_sp:
                        note_type_sp = 'half'
                    if 'quarter' in note_type_sp:
                        note_type_sp = 'quarter'
                    if 'eighth' in note_type_sp:
                        note_type_sp = 'eighth'
                    if 'sixteenth' in note_type_sp:
                        note_type_sp = '16th'
                    if 'thirty_second' in note_type_sp:
                        note_type_sp = '32rd'
                    break
            if count == 0:
                m.append(note.Note(note_name, type=note_type_sp))
            else:
                m.append(note.Note(note_name, type=note_type_sp, dots=count))
    stream1.write('midi', 'music/'+filename+'_piano.mid')
    os.popen("java -jar music/midi2wav.jar music/")


def note2chime(notestxt, filename):
    wav_path = 'func/ffmpeg/chimeNote/'
    musiclist = AudioSegment.empty()
    second_silence = AudioSegment.silent(duration=500)
    for i in notestxt:
        note_type, note_desc = split_notetxt(i)
        if note_type == 'note':
            note_wav = ''
            for j in note_desc:
                if j in ['C', 'D', 'E', 'F', 'G', 'A', 'B'] or j in ['3', '4', '5']:
                    note_wav += j
                elif j in ['1','2']:
                    note_wav += '3'
                elif j in ['6','7','8']:
                    note_wav += '5'
            file_wav = wav_path + note_wav + '_quarter.wav'
            sound = AudioSegment.from_file(file_wav, format='wav')
            musiclist += sound
        elif note_type == 'rest':
            musiclist += second_silence
        elif note_type == 'barline':
            musiclist += AudioSegment.silent(duration=250)
    musiclist.export('music/'+filename+'_chime.wav', format='wav')
