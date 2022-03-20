itn.far: itn.grm inumerals.far idate.far icurrency.far itime.far iphone.far
	thraxcompiler --input_grammar=$< --output_far=$@

inumerals.far: inumerals.grm ../common/lib/byte.far ../common/lib/chars.far dictionaries/number/ordinal/one_ordinals_alone.dict dictionaries/number/ordinal/1digit_ordinal.dict dictionaries/number/ordinal/1digit_ordinals.dict dictionaries/number/ordinal/1digit_ordinal_alone.dict dictionaries/number/ordinal/1digit_ordinals_alone.dict dictionaries/number/cardinal/1digit_cardinal.dict dictionaries/number/cardinal/1digit_alone_cardinal.dict dictionaries/number/ordinal/2digit_ordinal.dict dictionaries/number/ordinal/2digit_ordinals.dict dictionaries/number/cardinal/2digit_cardinal.dict dictionaries/number/cardinal/2digit_cardinal_alone.dict
	thraxcompiler --input_grammar=$< --output_far=$@

../common/lib/byte.far: ../common/lib/byte.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

../common/lib/chars.far: ../common/lib/chars.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

idate.far: idate.grm ../common/lib/byte.far dictionaries/date/months.dict
	thraxcompiler --input_grammar=$< --output_far=$@

icurrency.far: icurrency.grm ../common/lib/byte.far ../common/lib/chars.far dictionaries/currency/currency.dict dictionaries/currency/cryptocurrency.dict dictionaries/currency/cents.dict
	thraxcompiler --input_grammar=$< --output_far=$@

itime.far: itime.grm ../common/lib/byte.far ../common/lib/chars.far dictionaries/time/weekdays.dict dictionaries/time/months.dict dictionaries/time/normalization_context_left.cont dictionaries/time/forbidden_verb_left.dict dictionaries/time/forbidden_noun_left.dict dictionaries/time/forbidden_noun_right.dict dictionaries/time/last_forbidden_word.dict dictionaries/time/forbidden_sentence_left.sent dictionaries/time/forbidden_sentence_right.sent
	thraxcompiler --input_grammar=$< --output_far=$@

iphone.far: iphone.grm ../common/lib/byte.far ../common/lib/chars.far
	thraxcompiler --input_grammar=$< --output_far=$@

clean:
	rm -f inumerals.far idate.far icurrency.far itime.far iphone.far ../common/lib/byte.far ../common/lib/chars.far
