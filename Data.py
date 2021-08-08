import Audio
import Visual

# calculate the mean and variance of the data within the last pre-set seconds
def process_data(counter_head, counter_tail, all_visual_keypoints, all_audio_keypoints, COORDINATES):
    # process audio data
    output_data = []
    freq_mean, freq_var = Audio.process_audio('frequency', counter_head, counter_tail, all_audio_keypoints)
    amp_mean, amp_var = Audio.process_audio('amplitude', counter_head, counter_tail, all_audio_keypoints)
    # output_audio.append([freq_mean, freq_var, amp_mean, amp_var])
    output_data.append(freq_mean)
    output_data.append(freq_var)
    output_data.append(amp_mean)
    output_data.append(amp_var)

    # process visual data
    Visual.process_visual(33, 'pose', output_data, all_visual_keypoints, counter_head, counter_tail, COORDINATES)
    Visual.process_visual(468, 'face', output_data, all_visual_keypoints, counter_head, counter_tail, COORDINATES)
    Visual.process_visual(21, 'left_hand', output_data, all_visual_keypoints, counter_head, counter_tail, COORDINATES)
    Visual.process_visual(21, 'right_hand', output_data, all_visual_keypoints, counter_head, counter_tail, COORDINATES)
    
    return output_data