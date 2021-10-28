# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:39:25 2021

@author: user
"""

import numpy as np
import mido
from collections import namedtuple

Event = namedtuple("Event", "time duration pitch velocity")
ChordifiedEvent = namedtuple("Event", "time duration pitches onset velocities")

# Event = namedtuple("Event", "time duration data")
# ChordifiedEvent = namedtuple("Event", "time duration data onset")

def my_read_midi_mido(file):
    """
    Read notes with onset and duration from MIDI file. Time is specified in beats.
    :param file: path to MIDI file
    :return: sorted list of pitch Events
    """
    mid = mido.MidiFile(file)
    piece = []
    ticks_per_beat = mid.ticks_per_beat
    for track_id, t in enumerate(mid.tracks):
        time = 0
        track = []
        end_of_track = False
        active_notes = {}
        for msg in t:
            time += msg.time / ticks_per_beat
            if msg.type == 'end_of_track':
                # check for end of track
                end_of_track = True
            else:
                if end_of_track:
                    # raise if events occur after the end of track
                    raise ValueError("Received message after end of track")
                elif msg.type == 'note_on' or msg.type == 'note_off':
                    # only read note events
                    note = (msg.note, msg.channel)
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # note onset
                        if note in active_notes:
                            raise ValueError(f"{note} already active")
                        else:
                            active_notes[note] = (time, msg.velocity)
                    else:
                        # that is: msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
                        # note offset
                        if note in active_notes:
                            onset_time = active_notes[note][0]
                            note_duration = time - active_notes[note][0]
                            note_velocity = active_notes[note][1]
                            # append to track
                            track.append(Event(time=onset_time,
                                               duration=note_duration,
                                               pitch=msg.note,
                                               velocity=note_velocity))
                            del active_notes[note]
                        else:
                            raise ValueError(f"{note} not active (time={time}, msg.type={msg.type}, "
                                             f"msg.velocity={msg.velocity})")
        piece += track
    return list(sorted(piece, key=lambda x: x.time))
# end my_read_midi_mido

def empty_tabready_event_dict():
    return {
        'duration': -1.0,
        'onset_piece': -1.0,
        'onset_measure': -1.0,
        'pitches': []
    }
# end empty_tabready_event_dict
def empty_tabready_pitch_dict():
    return {
        'string': -1,
        'fret': -1,
        'pitch': -1,
        'velocity': -1,
        'duration_percentage': -1
    }
# end empty_tabready_pitch_dict

def onsetEvents2tabreadyEvents(events, parts_per_quarter=True):
    '''
    standard midi is 480 parts per quarter
    '''
    ppq_mult = 480
    if not parts_per_quarter:
        ppq_mult = 1
    tabreadyEvents = []
    for e in events:
        tmpEvent = empty_tabready_event_dict()
        tmpEvent['duration'] = e.duration*ppq_mult if not parts_per_quarter else np.round(e.duration*ppq_mult).astype(int)
        tmpEvent['onset_piece'] = e.time*ppq_mult  if not parts_per_quarter else np.round(e.time*ppq_mult).astype(int)
        tmpEvent['pitches'] = []
        for p in list(e.pitch):
            tmpPitch = empty_tabready_pitch_dict()
            tmpPitch['pitch'] = p
            tmpPitch['velocity'] = e.velocity
            tmpPitch['duration_percentage'] = 1.0
            tmpEvent['pitches'].append( tmpPitch )
        tabreadyEvents.append(tmpEvent)
    return tabreadyEvents
# end onsetEvents2tabreadyEvents

def my_chordify(piece):
    """
    Create time bins at note events (on- or offset). For each bin create a set of notes that are on.
    :param piece: List of pitch Events
    :return: list of Events (with start time and duration) with note sets
    """
    '''
    MAX: duration events is not pianoroll with duration, but all active events,
    regardless of their onset. That is, a chordified version of the score, where
    durations are NOT represented in time. Onset events, is the chordified version
    but only for events that are getting activated (have onset), not events that
    have been active from previous activation. Therefore, the piano roll produced
    by those matrices does not correspond to a time-accurate pianoroll, only to
    an active-notes or onset-accurate piano roll.
    '''
    # create dictionary with time on- and offsets and events starting at a certain onset
    event_dict = {}
    onset_dict = {}
    for e in piece:
        # add onset and offset time slot
        if e.time not in event_dict:
            event_dict[e.time] = set()
            onset_dict[e.time] = set()
        if e.time + e.duration not in event_dict:
            event_dict[e.time + e.duration] = set()
            onset_dict[e.time + e.duration] = set()
        # add event to onset time slot
        event_dict[e.time].add(e)
        onset_dict[e.time].add(e)
        # print('event_dict[' + str(e.time) + ']: ', event_dict[e.time])
    # turn dict into ordered list of time-events pairs
    event_list = list(sorted(event_dict.items(), key=lambda item: item[0]))
    onset_list = list(sorted(onset_dict.items(), key=lambda item: item[0]))
    # take care of events that extend over multiple time slots
    active_events = set()
    for time, events in event_list:
        # from the active events (that started earlier) only keep those that reach into current time slot
        active_events = set(event for event in active_events if event.time + event.duration > time)
        # add these events to the current time slot
        events |= active_events
        # remember events that start with this slot to possibly add them in later slots
        active_events |= events
    # the last element should not contain any events as it was only created because the last event(s) ended at that time
    if event_list[-1][1]:
        raise ValueError(f"The last time slot should be empty but it contains: '{event_list[-1][1]}'. "
                         f"This is a bug (maybe due to floating point arithmetics?)")
    # turn dict into an ordered list of events with correct durations and combined event data
    duration_events = [Event(time=time, 
                             duration=next_time - time, 
                             pitch=frozenset([e.pitch for e in events]), 
                             velocity=list(events)[0].velocity if len(list(events)) else -1)
            for (time, events), (next_time, next_events) in zip(event_list, event_list[1:])]
    onset_events = [Event(time=time, 
                          duration=next_time - time, 
                          pitch=frozenset([e.pitch for e in events]),
                          velocity=list(events)[0].velocity if len(list(events)) else -1)
            for (time, events), (next_time, next_events) in zip(onset_list, onset_list[1:])]
    return duration_events, onset_events
# end my_chordify


def my_piano_roll(c, min_pitch=None, max_pitch=None, return_range=False,
                  return_durations=False, return_onsets=False):
    # read piece and chordify
    # chordified = chordify(reader(file))
    chordified = c
    assert len(chordified) > 0, "this piece seems to be empty"
    # get all occuring pitches
    all_pitches = frozenset.union(*[event.data for event in chordified])
    # get actual minimum and maximum pitch
    actual_min_pitch = min(all_pitches)
    actual_max_pitch = max(all_pitches)
    # check against requested values
    if min_pitch is None:
        min_pitch = actual_min_pitch
    else:
        if actual_min_pitch < min_pitch:
            raise ValueError(f"actual minimum pitch ({actual_min_pitch}) is smaller than requested value ({min_pitch}")
    if max_pitch is None:
        max_pitch = actual_max_pitch
    else:
        if actual_max_pitch > max_pitch:
            raise ValueError(f"actual maximum pitch ({actual_max_pitch}) is greater than requested value ({max_pitch}")
    assert max_pitch >= min_pitch  # safety check
    # allocate numpy array of appropriate size
    roll = np.zeros((len(chordified), max_pitch - min_pitch + 1), dtype=np.bool)
    # onsets
    ons = np.zeros((len(chordified), max_pitch - min_pitch + 1), dtype=np.bool)
    # set multiple-hot for all time slices
    for time_idx, event in enumerate(chordified):
        if len(list(event.data)) > 0:
            roll[time_idx, np.array(list(event.data)) - min_pitch] = True
    # construct return tuple
    ret = (roll,)
    if return_range:
        ret = ret + (np.arange(min_pitch, max_pitch + 1),)
    if return_durations:
        ret = ret + (np.array([event.duration for event in chordified]),)
    # return
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
# end my_piano_roll