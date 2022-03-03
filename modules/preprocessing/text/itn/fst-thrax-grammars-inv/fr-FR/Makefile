itn.far: itn.grm inumerals.far idate.far
	thraxcompiler --input_grammar=$< --output_far=$@

inumerals.far: inumerals.grm ../common/lib/byte.far ../common/lib/chars.far dictionaries/number/ordinal/one_ordinals_alone.dict dictionaries/number/ordinal/1digit_ordinal.dict dictionaries/number/ordinal/1digit_ordinals.dict dictionaries/number/ordinal/1digit_ordinal_alone.dict dictionaries/number/ordinal/1digit_ordinals_alone.dict dictionaries/number/cardinal/1digit_cardinal.dict dictionaries/number/cardinal/1digit_alone_cardinal.dict dictionaries/number/ordinal/2digit_ordinal.dict dictionaries/number/ordinal/2digit_ordinals.dict dictionaries/number/cardinal/2digit_cardinal.dict dictionaries/number/cardinal/2digit_cardinal_alone.dict
	thraxcompiler --input_grammar=$< --output_far=$@

../common/lib/byte.far: ../common/lib/byte.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

../common/lib/chars.far: ../common/lib/chars.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

idate.far: idate.grm ../common/lib/byte.far dictionaries/date/months.dict
	thraxcompiler --input_grammar=$< --output_far=$@

clean:
	rm -f inumerals.far idate.far ../common/lib/byte.far ../common/lib/chars.far
