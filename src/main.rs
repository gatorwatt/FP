// Copyright 2023 Nicholas Teague
// released under BSD-3-Clause license

// subnormal support for floating operations pending
// (for now treating zero as a special encoding)
// special encoding support for mixed precision pending
// unsigned floating point not yet implemented
// signed integer not yet implemented
// current inspection of special encodings is a little hacky
// future update may do this without comparison to f32
// just need to think about cleanest way to implement

// I think it goes without saying that each of the subcomponents herein
// probably have the potential for further refinement and optimization
// the goal here is to establish a foundation for operational primitives
// which may be further refined going forward if there is a need

// note that currently using u8 for a lot of bitwidth variables
// which might limit this for extremely high precision applicaitons
// may reconsider basis to usize down the road

// note that a practice I have adopted in architecting
// is to generally avoid global variables
// they tend to be sources of edge cases and etc
// which is well suited for rust conventions
// and will probably benefit considerations like parallelization down the road

// methods do not currently implement flagging for edge cases
// just rely on propagation of special encodings

// right now don't have a really clean mechanism for overflow / underflow, pending further thought


// #![allow(warnings)]

// used to sample between set of values (0/1)
use rand::Rng;
use rand::thread_rng;

// // used to sample from a distribution
// // (currently only used for validations)
// use rand::prelude::*;
// use rand_distr::{Normal, Uniform};

// // used to plot histogram
// use plotlib::page::Page;
// use plotlib::repr::{Histogram, HistogramBins};
// use plotlib::view::ContinuousView;

// #[derive(Debug)] // this adds printing support for struct
// #[derive(Clone)] // don't remember what this was for

// Table of Contents for main(.)
// ### [a] initialize configuration parameters ###  
// ### [b] Sample or Specify a target encoding ###
// ### [c] Sample or Specify a target float ###
// ### [d] Configurable 754 benchmarking ###
// ### [e] Posits benchmarking ###
// ### [f] Operations benchmarking ###

// Table of Contents for support functions
// *** [0] Benchmarking Functions ***
// *** [1] Configurable 754 Functions ***
// *** [2] Posits Functions ***
// *** [3] Methods to Sample Encodings ***
// *** [4] bool vector support functions ***
// *** [5] bitwise operators ***
// *** [6] integer operations ***
// *** [7] floating point operations ***
// *** [8] mixed precision support ***

// ________________________________________
// struct data structures
// SpecialEncodings: nan, infinity, neginfinity
// each to populated with vector of reserved encodings for special values
// includes methods to inspect a value and return a vector
// and vica versa
// check_for_special_value(.) and check_for_reserved_encoding(.)
// ** pending: check_for_reserved_encoding_mixedprecision
// PositSupport: based on a particular configuration, 
// prepropulates values for use to translate floats to encodings

struct SpecialEncodings {
  nan : Vec<bool>,
  nan_equals_nan : bool,
  infinity : Vec<bool>,
  neginfinity : Vec<bool>,
  zero : Vec<bool>,
  negzero : Vec<bool>, // the negzero treatment needs further thought since may be using this for nan basis, pending
  stochastic_rounding : bool,
  
  // negzero : Vec<bool>, // not currently inspected
}

struct PositSupport {
  expwidth : u8,
  k_multiplier : f64, // pending, need to add support to get_posits_value
  k_set : Vec<f64>,
  u : f64,
  u_to_k_set : Vec<f64>,
  cons_set : Vec<u32>,
  regime_set : Vec<u8>,
  two_to_cons_set : Vec<u32>,
  u_to_k_times_two_to_cons_set : Vec<f64>,
  max_mantissa_set : Vec<f64>,
  max_abs_set : Vec<f64>,
  min_abs_set : Vec<f64>,
}

// we'll have static methods for the SpecialEncodings structure
// that tests a float for special values
// and when found returns the corresponding vector
// and then a similar method in reverse direction

impl SpecialEncodings {
  
  fn check_for_special_value(&self, value:f32) -> Vec<bool> {
    // tests a value to determinei if is a special value (NaN, INF, NEG_INF)
    // if so returns the specified vector 
    // from this instance of struct SpecialEncodings
    
    let empty_encoding:Vec<bool> = vec![];

    let encoding:&Vec<bool> = if f32::is_nan(value) {
      &self.nan
    } else if value == f32::INFINITY {
      &self.infinity
    } else if value == f32::NEG_INFINITY {
      &self.neginfinity
    } else if value == 0. {
      &self.zero
    } else if value == -0. {
      &self.negzero
    } else {
      &empty_encoding
    };
    
    return encoding.to_vec();
  }

  fn check_for_reserved_encoding(&self, vector:&Vec<bool>) -> f32 {
    // tests a value to determinei if is a reserved vector
    // from this instance of struct SpecialEncodings
    // if so returns the associated special value (NaN, INF, NEG_INF)
    // otherwise returns as the float 1.0f32 as a plug value
    
    // default assumes no special encoding
    let mut value:f32 = 1.0f32;

    if vector == &self.nan {
      value = f32::NAN;
    } else if vector == &self.infinity {
      value = f32::INFINITY;
    } else if vector == &self.neginfinity {
      value = f32::NEG_INFINITY;
    } else if vector == &self.zero {
      value = 0.;
    } else if vector == &self.negzero {
      value = -0.;
    }
      
    return value;
  }

  fn convert_reserved_encodings(&self, expwidth_input_basis:u8, expwidth_output_basis:u8, bitwidth_output_basis:u8) -> SpecialEncodings {
  // for cases where original specification of reserved encodings
  // is to be used to detect special encodings in an alternate basis
  // (meaning alternate expwidth / bitwidth basis)
  // this function can be used to output a converted set of special encodings
  // which will rely on a few logic tests 
  // (e.g. all false in one precision can be converted to all false in alternate precision)
  // details of conversion basis documented in comments below
  // for setups with a large number of special encodings this might not scale
  // however for small number of encodings in FP8 conventions this is doable
  // nan : Vec<bool>,
  // infinity : Vec<bool>,
  // neginfinity : Vec<bool>,
  // zero : Vec<bool>,
  // negzero : Vec<bool>,

  // _______________________
  // we assume that a typical special encoding specification
  // might resemble following

  // // SPECIAL ENCODINGS ON SCENARIO, FP8 CONVENTIONS
  // // conventions for inf at saturation and nan at neg zero with bitwidth=8
  // let special_encodings = SpecialEncodings {
  //   nan : vec![true, false, false, false, false, false, false, false], // this is otherwise negative zero
  //   infinity : vec![false, true, true, true, true, true, true, true], // this is otherwise saturation
  //   neginfinity : vec![true, true, true, true, true, true, true, true], // this is otherwise negative saturation
  //   zero : vec![false, false, false, false, false, false, false, false],
  //   negzero : Vec::<bool>::new(), // not currently inspected
  //  stochastic_rounding : false,
  // _______________________

  // so let's just walkt through each detectable pattern

    fn match_encoding_pattern(orig_vector:&Vec<bool>, expwidth_input_basis:u8, expwidth_output_basis:u8, bitwidth_output_basis:u8) -> Vec<bool> {
      // support function for convert_reserved_encodings
      // receives a vector in prior basis expwidth_input_basis
      // and converts to basis of expwidth_output_basis / bitwidth_output_basis
  
      // initialize output
      let mut output_vector:Vec<bool> = vec![];
  
      // here are a few patterns we may see
  
      // no specification scenario
      if orig_vector.len() == 0 {
        return output_vector;
      } else if !orig_vector.contains(&true) {
        // all false scenario
        output_vector = vec![false; bitwidth_output_basis as usize];
      } else if !orig_vector.contains(&false) {
        // all true scenario
        output_vector = vec![true; bitwidth_output_basis as usize];
      } else if orig_vector[0] && !orig_vector[1..].contains(&true) {
        // activated sign bit and rest false scenario
        output_vector = vec![ vec![true], vec![false; bitwidth_output_basis as usize - 1]].concat();
      } else if !orig_vector[0] && !orig_vector[1..].contains(&false) {
        // null sign bit and rest true scenario
        output_vector = vec![ vec![false], vec![true; bitwidth_output_basis as usize - 1]].concat();
      } else if !orig_vector[1..(expwidth_input_basis + 1) as usize].contains(&false) {
        // this is the all null exponent scenarios
        // (e.g. for subnormal numbers in some conventions)
        if !orig_vector[(expwidth_input_basis + 1) as usize..].contains(&false) {
          // for "subnormal" exp with all false
          // construct output with matched sign bit
          // all null exp
          // and if mantissa was all true pad out with true
          output_vector = vec![vec![orig_vector[0]], vec![false; expwidth_output_basis as usize], vec![true; (bitwidth_output_basis - expwidth_output_basis - 1) as usize]].concat();
        
        }
      }
  
        // (this one was given me a headache), 
        // } else {
        //   // else pad out input mantissa with false
        //   // (this might be debatable since input mantissa might have different expwidth basis)
        //   // just trying to pick a convention
        //   let expwidth_delta:i16 = expwidth_ouput_basis as i16 - expwidth_input_basis as i16;
          
        //   let mut output_mantissa:Vec<bool> = vec![orig_vector[(1+expwidth_input_basis) as usize..], vec![false; ((bitwidth_output_basis - 1 - expwidth_ouput_basis)]].concat();
          
        //   output_vector = vec![vec![orig_vector[0]], vec![false; expwidth_output_basis as usize], vec![true; (bitwidth_output_basis - expwidth_output_basis - 1) as usize]].concat();
        // }
        
      return output_vector;
    }
  
    // let special_encodings = SpecialEncodings {
    //   nan : vec![true, false, false, false, false, false, false, false], // this is otherwise negative zero
    //   infinity : vec![false, true, true, true, true, true, true, true], // this is otherwise saturation
    //   neginfinity : vec![true, true, true, true, true, true, true, true], // this is otherwise negative saturation
    //   zero : vec![false, false, false, false, false, false, false, false],
    //   negzero : Vec::<bool>::new(), // not currently inspected
    //  stochastic_rounding : false,
  
    let returned_special_encodings = SpecialEncodings {
  
      nan : match_encoding_pattern(&self.nan, expwidth_input_basis, expwidth_output_basis, bitwidth_output_basis),
      nan_equals_nan : *&self.nan_equals_nan,
      infinity : match_encoding_pattern(&self.infinity, expwidth_input_basis, expwidth_output_basis, bitwidth_output_basis),
      neginfinity : match_encoding_pattern(&self.neginfinity, expwidth_input_basis, expwidth_output_basis, bitwidth_output_basis),
      zero : match_encoding_pattern(&self.zero, expwidth_input_basis, expwidth_output_basis, bitwidth_output_basis),
      negzero : match_encoding_pattern(&self.negzero, expwidth_input_basis, expwidth_output_basis, bitwidth_output_basis),
      stochastic_rounding : false,
  
    };
  
    return returned_special_encodings;
  }
}


// ___________________________________
// begin demonstration

fn main() {

  //__________________________________________________________
  
  
  
  // ### [a] initialize configuration parameters ###  


  //_________________
  //values serving as basis for default configuration
  let mut bitwidth:u8 = 8;
  let mut expwidth:u8 = 4;

  // bitwidth = 12;
  // expwidth = 6;


  

  //_________________
  //default special encoding configurations

  // SPECIAL ENCODINGS ON SCENARIO, FP8 CONVENTIONS
  // conventions for inf at saturation and nan at neg zero with bitwidth=8
  let special_encodings = SpecialEncodings {
    
    nan : vec![true, false, false, false, false, false, false, false], // this is otherwise negative zero
    
    nan_equals_nan : false,
    
    infinity : vec![false, true, true, true, true, true, true, true], // this is otherwise saturation
    
    neginfinity : vec![true, true, true, true, true, true, true, true], // this is otherwise negative saturation
    
    zero : vec![false, false, false, false, false, false, false, false],
    
    negzero : Vec::<bool>::new(), // not currently inspected
    
    stochastic_rounding : false,
  };
  

  // SPECIAL ENCODINGS OFF SCENARIO
  // we'll have default special encodings of empty vector for when not specified
  let special_encodings_off = SpecialEncodings {
    nan : Vec::<bool>::new(), // equivalent to initializing as vec![] but assigns a type
    nan_equals_nan : true,
    infinity : Vec::<bool>::new(),
    neginfinity : Vec::<bool>::new(),
    zero : Vec::<bool>::new(),
    negzero : Vec::<bool>::new(), // not currently inspected
   stochastic_rounding : false,
  };


  //__________________________________________________________
  // ### [b] Sample or Specify a target encoding ###

  // // we can randomly sample an array of bool to derive value
  // let sampled_array : Vec<bool> = sample_bool_array(bitwidth);
  // println!("");
  // println!("{:?}", sampled_array);

  // // // alternatively can define a specific array to validate value
  // // // let sampled_array : Vec<bool> = vec![true, false, false, true, true, false, false, true];

  // let recovered_value = get_value(&sampled_array, expwidth, &special_encodings);
  // println!("");
  // println!("{:?}", sampled_array);
  // println!("recovered_value = {}", recovered_value);


  // // demonsrtate information retention comparison
  // let encoded_float:Vec<bool> = encode_float(recovered_value as f64, bitwidth, expwidth, &special_encodings);
  // println!();
  // println!("target_number = {}", recovered_value);
  // println!("orig encoded_float {:?}", sampled_array);
  // println!("recovered encoded_float = {:?}", encoded_float);
  // println!("original == recovered is {}", sampled_array == encoded_float);



  //_______________________________________________________
  // ### [c] Sample or Specify a target float ###
  
  // println!("");
  // println!("");


  // // 1
  // println!("");
  // println!("floats for validation can be sampled or specified:");
  // // if wanted to sample from a Gaussian
  // let normal = Normal::new(0.0, 1.0).unwrap();
  // let target_number:f64 = normal.sample(&mut rand::thread_rng());
  // println!("gaussian sample = {}", target_number);

  // // 2
  // // if wanted to sample from a Uniform
  // let uniform = Uniform::new::<f64, f64>(-1.0, 1.0);
  // let target_number:f64 = uniform.sample(&mut rand::thread_rng());
  // println!("uniform sample = {}", target_number);
  
  // // 3
  // // or for a specific target number can specify here
  // // let target_number:f64 = f64::NAN;
  // let target_number:f64 = -0.241;
  // println!("manually specified float = {}", target_number);

  // // 4
  // // demonsrtate information retention comparison
  // let encoded_float:Vec<bool> = encode_float(target_number, bitwidth, expwidth, &special_encodings);
  // println!();
  // println!("target_number = {}", target_number);
  // println!("encoded_float = {:?}", encoded_float);

  // // 5
  // let recovered_value = get_value(&encoded_float, expwidth, &special_encodings);
  // println!("");
  // println!("target_number = {}", target_number);
  // println!("recovered_value = {}", recovered_value);
  

  
  // _____________________________________________________
  // ### [d] Configurable 754 benchmarking ###

  // let sampled_array : Vec<bool> = sample_bool_array(bitwidth);
  // println!("");
  // println!("Sample a full encoding of bool entries:");
  // println!("{:?}", sampled_array);
  // //this is based on generic conventions resembling 754
  // println!("Convert to float:");
  // let value = get_value(&sampled_array, expwidth, &special_encodings);
  // println!("value = {}", value);
  // // println!("{:?}", sampled_array);



  // // 6
  // // this shows all encodings for a given bit width
  // let all_encodings:Vec<bool> = get_all_encodings(bitwidth);
  // println!("");
  // println!("{:?}", all_encodings);
  // println!("");

  // // 7
  // // this returns all values associated with the configurable 754 convention
  // let all_values:Vec<f32> = get_all_values(&all_encodings, bitwidth, expwidth, &special_encodings, "754");
  // println!("");
  // println!("all 754 values {:?}", all_values);
  // println!("");


  // // 8
  // // this returns a tuple in form:
  // // (max_value, max_encoding, min_value, min_encoding, zero_value, zero_encoding, negzero_value, negzero_encoding)
  // println!("calculate max / min for an encoding (shown saturation values without special encodings)");
  // let encodings_tuple = get_saturation_and_zero_encodings(bitwidth, expwidth, &special_encodings_off, "754");
  // println!("{:?}", encodings_tuple);



  // // 9
  // // this encodes and reverts all values 
  // // which validates consistency
  // println!("");
  // println!("encode and revert all values to confirm consistency:");
  // let all_revert_valresult:bool = validate_all_values(&all_values, bitwidth, expwidth, &special_encodings);
  // println!("all_revert_valresult = {}", all_revert_valresult);




  // //________________________________________________________
  // // ### [e] Posits benchmarking ###

  // posits support currently experimental

  //this is based on posits conventions
  // note that in posits convention there is an additional degree of freedom
  // for bits dedicated to the exponent
  // so we'll interpret expwidth as the maximum number of bits applied to the exponent

  // let bitwidth:u8 = 8;
  // let expwidth:u8 = 3;


  // println!("start validate");
  // let posit_config = populate_posit_config(bitwidth, expwidth, 1.0);

  // let sampled_array : Vec<bool> = sample_bool_array(bitwidth);
  // println!("");
  // println!("{:?}", sampled_array);
  // let posits_value = get_posits_value(&sampled_array, bitwidth, expwidth);
  // println!("{:?}", sampled_array);
  // println!("posits value = {}", posits_value);

  
  // let recovered_array:Vec<bool> = encode_posits_float(posits_value as f64, bitwidth, expwidth, &posit_config);

  // println!("recovered_array");
  // println!("{:?}", recovered_array);

  // let reverted_posits_value = get_posits_value(&recovered_array, bitwidth, expwidth);
  // println!("reverted_posits_value = {}", reverted_posits_value);


  // // this shows all encodings for a given bit width
  // let all_encodings:Vec<bool> = get_all_encodings(bitwidth);
  // println!("{:?}", all_encodings);

  // // this returns all values associated with the posits convention
  // // (needs validation)
  // let all_posits_values:Vec<f32> = get_all_values(&all_encodings, bitwidth, expwidth, &special_encodings, "posits");
  // println!("all posits values {:?}", all_posits_values);


  // // can plot a rudimentary ASCI histogram
  // // using plotlib crate
  // // (requires turning off special encodings)
  // let mut data : Vec<f64> = vec![];
  // // for entry in all_values {
  // for entry in all_posits_values {
  //   data.push(entry as f64);
  // }
  // let h = Histogram::from_slice(&data, HistogramBins::Count(10));
  // let v = ContinuousView::new().add(h);
  // println!("");
  // println!("{}", Page::single(&v).dimensions(60, 15).to_text().unwrap());

  
  
  //________________________________________________________
  // ### [f] Operations benchmarking ###

  let mut sampled_array_1 : Vec<bool> = vec![];
  let mut sampled_array_2 : Vec<bool> = vec![];
  let mut sampled_array_3 : Vec<bool> = vec![];
  
  let mut recovered_value_1 : f32 = 0.;
  let mut recovered_value_2 : f32 = 0.;
  let mut recovered_value_3 : f32 = 0.;

  let mut reverted_vector : f32 = 0.;

  let mut f32_comparison : f32 = 0.;
  let mut recovered_array_from_f32 : Vec<bool> = vec![];

  let mut recovered_int_1 : u32 = 0;
  let mut recovered_int_2 : u32 = 0;
  let mut recovered_int_3 : u32 = 0;
  
  let mut returned_bool:bool = false;
  
  // // validate_bitwise_and_integer_operations();

  // benchmark_bitwise_ops(bitwidth);

  // // _______________

  // // integer_convert(.) - for use to validate the integer arithmetic operations
  // // integer_revert(.) - translates a u32 to Vec<bool>
  // // integer_add(.)
  // // integer_increment(.) - increments vector encoding by 1
  // // integer_decrement(.) - decrements vector encoding by 1
  // // integer_subtract(.)
  // // integer_multiply(.)
  // // integer_greaterthan(.)
  // // integer_divide(.)

  // benchmark_integer_ops(bitwidth);

  // // _______________

  // // fp_greaterthan(.)
  // // fp_add(.)
  // // fp_add_variable_exp(.)
  // // integer_add_extended(.)
  // // fp_subtract(.)
  // // fp_multiply(.)
  // // fp_divide(.)
  // // fp_compare(.)
  // // fp_match_precision(.)
  // // fp_fma(.)
  // // fp_dot(.)



  benchmark_fp_ops(bitwidth, expwidth, &special_encodings);
  

  println!("_______________");
  println!("");
  println!("Additional operations implemented, pending further quality control:");
  println!("");
  // println!("- fp_multiply");
  println!("- fp_divide (doesn't yet support arbitrary precision");
  // println!("- fp_compare");
  // println!("- fp_match_precision");
  println!("- nearest or stochastic rounding");
  // println!("- fp_match_FMA");
  // println!("- fp_match_dot");
  println!("- posits format conversions");
  // println!("- special encodings support for some of operations pending");
  println!("- special encodings support for mixed precision operations pending");
  println!("- flags / saturation output for fp overflow / underflow pending");
  println!(" ");

  
}

// // ___________________________________________________________
// // *** [0] Benchmarking Functions ***
// // following intended as extensions:
// // - compare information retention for a specific float
// // - derive aggregate metric for information retention in set of sampled floats
// // - information retention comparisons between alternate encoding types
// // - generate scatter plot of all encodings
// // - identify a model approximate distribution based on densities of an encoding type
// // - derive signal to noise ratio for encoding type in comparison to a target distribution

// benchmark_bitwise_ops
// benchmark_integer_ops
// benchmark_fp_ops

fn benchmark_bitwise_ops(bitwidth:u8) {

  let mut sampled_array_1 : Vec<bool> = vec![];
  let mut sampled_array_2 : Vec<bool> = vec![];
  let mut sampled_array_3 : Vec<bool> = vec![];
  
  let mut recovered_value_1 : f32 = 0.;
  let mut recovered_value_2 : f32 = 0.;
  let mut recovered_value_3 : f32 = 0.;

  let mut reverted_vector : f32 = 0.;

  let mut f32_comparison : f32 = 0.;
  let mut recovered_array_from_f32 : Vec<bool> = vec![];

  let mut recovered_int_1 : u32 = 0;
  let mut recovered_int_2 : u32 = 0;
  let mut recovered_int_3 : u32 = 0;
  
  let mut returned_bool:bool = false;

  // validate_bitwise_and_integer_operations();

  // bitwise_and
  sampled_array_1 = sample_bool_array(bitwidth);
  sampled_array_2 = sample_bool_array(bitwidth);
  sampled_array_3 = bitwise_and(&sampled_array_1, &sampled_array_2);

  println!("");
  println!("________");
  println!("BITWISE AND");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: bitwise_and");
  println!("result: {:?}", sampled_array_3);
  println!("");

  // bitwise_or
  sampled_array_1 = sample_bool_array(bitwidth);
  sampled_array_2 = sample_bool_array(bitwidth);
  sampled_array_3 = bitwise_or(&sampled_array_1, &sampled_array_2);

  println!("");
  println!("________");
  println!("BITWISE OR");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: bitwise_and");
  println!("result: {:?}", sampled_array_3);
  println!("");

  // bitwise_xor
  sampled_array_1 = sample_bool_array(bitwidth);
  sampled_array_2 = sample_bool_array(bitwidth);
  sampled_array_3 = bitwise_xor(&sampled_array_1, &sampled_array_2);

  println!("");
  println!("________");
  println!("BITWISE XOR");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: bitwise_xor");
  println!("result: {:?}", sampled_array_3);
  println!("");

  // bitwise_not
  sampled_array_1 = sample_bool_array(bitwidth);
  // sampled_array_2 = sample_bool_array(bitwidth);
  sampled_array_3 = bitwise_not(&sampled_array_1);

  println!("");
  println!("________");
  println!("BITWISE NOT");
  println!("sampled vector 1 {:?}", sampled_array_1);
  // println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: bitwise_not");
  println!("result: {:?}", sampled_array_3);
  println!("");


  // bitwise_leftshift
  sampled_array_1 = sample_bool_array(bitwidth);
  // sampled_array_2 = sample_bool_array(bitwidth);
  sampled_array_3 = bitwise_leftshift(&sampled_array_1);

  println!("");
  println!("________");
  println!("BITWISE LEFTSHIT");
  println!("sampled vector 1 {:?}", sampled_array_1);
  // println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: bitwise_leftshift");
  println!("result: {:?}", sampled_array_3);
  println!("");


  // bitwise_rightshift
  sampled_array_1 = sample_bool_array(bitwidth);
  // sampled_array_2 = sample_bool_array(bitwidth);
  sampled_array_3 = bitwise_rightshift(&sampled_array_1);

  println!("");
  println!("________");
  println!("BITWISE RIGHTSHIFT");
  println!("sampled vector 1 {:?}", sampled_array_1);
  // println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: bitwise_rightshift");
  println!("result: {:?}", sampled_array_3);
  println!("");

  
}


fn benchmark_integer_ops(bitwidth:u8) {

  let mut sampled_array_1 : Vec<bool> = vec![];
  let mut sampled_array_2 : Vec<bool> = vec![];
  let mut sampled_array_3 : Vec<bool> = vec![];
  
  let mut recovered_value_1 : f32 = 0.;
  let mut recovered_value_2 : f32 = 0.;
  let mut recovered_value_3 : f32 = 0.;

  let mut reverted_vector : f32 = 0.;

  let mut f32_comparison : f32 = 0.;
  let mut recovered_array_from_f32 : Vec<bool> = vec![];

  let mut recovered_int_1 : u32 = 0;
  let mut recovered_int_2 : u32 = 0;
  let mut recovered_int_3 : u32 = 0;
  
  let mut returned_bool:bool = false;



  // integer_add
  sampled_array_1 = sample_bool_array(bitwidth);
  sampled_array_2 = sample_bool_array(bitwidth);
  recovered_int_1 = integer_convert(&sampled_array_1);
  recovered_int_2 = integer_convert(&sampled_array_2);
  sampled_array_3 = integer_add(&sampled_array_1, &sampled_array_2);
  recovered_int_3 = integer_convert(&sampled_array_3);


  println!("");
  println!("________");
  println!("INT ADD");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: integer_add");
  println!("returned vector: {:?}", sampled_array_3);
  println!("");
  println!("{} + {} = {}", recovered_int_1, recovered_int_2, recovered_int_3);
  println!("");


  // integer_subtract
  sampled_array_1 = sample_bool_array(bitwidth);
  sampled_array_2 = sample_bool_array(bitwidth);
  recovered_int_1 = integer_convert(&sampled_array_1);
  recovered_int_2 = integer_convert(&sampled_array_2);
  sampled_array_3 = integer_subtract(&sampled_array_1, &sampled_array_2);
  recovered_int_3 = integer_convert(&sampled_array_3);


  println!("");
  println!("________");
  println!("INT SUBTRACT");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: integer_subtract");
  println!("returned vector: {:?}", sampled_array_3);
  println!("");
  println!("{} - {} = {}", recovered_int_1, recovered_int_2, recovered_int_3);
  println!("");


  // integer_multiply
  sampled_array_1 = sample_bool_array(bitwidth);
  // sampled_array_2 = sample_bool_array(bitwidth);

  // multiply may result in overflow for fixed integer width
  // so here applying arbitrary small second value = 3
  sampled_array_2 = vec![false, false, false, false, false, false, true, true];
  let mut sampled_array_2_len = sampled_array_2.len();
  if sampled_array_2_len > bitwidth as usize {
    for i in 0..(sampled_array_2_len - bitwidth as usize) {
      sampled_array_2.remove(0);
    }
  } else if sampled_array_2_len < bitwidth as usize {
    for i in 0..(bitwidth as usize - sampled_array_2_len ) {
      sampled_array_2.insert(0, false);
    }
  }
  
  recovered_int_1 = integer_convert(&sampled_array_1);
  recovered_int_2 = integer_convert(&sampled_array_2);
  sampled_array_3 = integer_multiply(&sampled_array_1, &sampled_array_2);
  recovered_int_3 = integer_convert(&sampled_array_3);


  println!("");
  println!("________");
  println!("INT MULTIPLY");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: integer_multiply");
  println!("returned vector: {:?}", sampled_array_3);
  println!("");
  println!("{} * {} = {}", recovered_int_1, recovered_int_2, recovered_int_3);
  println!("");



  // integer_divide
  sampled_array_1 = sample_bool_array(bitwidth);
  sampled_array_2 = sample_bool_array(bitwidth);


  // can set first few bits in sampled_array_2 as false
  // to promote outputs > 1\
  for i in 0..(bitwidth as usize /2 as usize) {
    sampled_array_2[i] = false;
  }

  // // divide may result in underflow for fixed integer width
  // // so here applying arbitrary small second value = 3
  // sampled_array_2 = vec![false, false, false, false, false, false, true, true];
  // let mut sampled_array_2_len = sampled_array_2.len();
  // if sampled_array_2_len > bitwidth as usize {
  //   for _i in 0..(sampled_array_2_len - bitwidth as usize ) {
  //     sampled_array_2.remove(0);
  //   }
  // } else if sampled_array_2_len < bitwidth as usize {
  //   for _i in 0..(bitwidth as usize - sampled_array_2_len) {
  //     sampled_array_2.insert(0, false);
  //   }
  // }
  
  recovered_int_1 = integer_convert(&sampled_array_1);
  recovered_int_2 = integer_convert(&sampled_array_2);
  sampled_array_3 = integer_divide(&sampled_array_1, &sampled_array_2);
  recovered_int_3 = integer_convert(&sampled_array_3);


  println!("");
  println!("________");
  println!("INT DIVIDE");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: integer_divide");
  println!("returned vector: {:?}", sampled_array_3);
  println!("");
  println!("{} / {} = {}", recovered_int_1, recovered_int_2, recovered_int_3);
  println!("checksum = {}", (recovered_int_1 / recovered_int_2) as u32);
  println!("");

}


fn benchmark_fp_ops(bitwidth:u8, expwidth:u8, special_encodings:&SpecialEncodings) {



  let mut sampled_array_1 : Vec<bool> = vec![];
  let mut sampled_array_2 : Vec<bool> = vec![];
  let mut sampled_array_3 : Vec<bool> = vec![];
  
  let mut recovered_value_1 : f32 = 0.;
  let mut recovered_value_2 : f32 = 0.;
  let mut recovered_value_3 : f32 = 0.;

  let mut reverted_vector : f32 = 0.;

  let mut f32_comparison : f32 = 0.;
  let mut recovered_array_from_f32 : Vec<bool> = vec![];

  let mut recovered_int_1 : u32 = 0;
  let mut recovered_int_2 : u32 = 0;
  let mut recovered_int_3 : u32 = 0;
  
  let mut returned_bool:bool = false;
  

  // fp_greaterthan
  sampled_array_1 = sample_bool_array(bitwidth);
  sampled_array_2 = sample_bool_array(bitwidth);
  recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
  recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);
  returned_bool = fp_greaterthan(&sampled_array_1, &sampled_array_2, expwidth, &special_encodings);


  println!("");
  println!("________");
  println!("FP GREATERTHAN");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("operation: fp_greaterthan");
  println!("returned bool: {}", returned_bool);
  println!("");
  println!("{} >= {} = {}", recovered_value_1, recovered_value_2, returned_bool);
  println!("");



  for i in 0..1000 {
    
    // fp_add
    sampled_array_1 = sample_bool_array(bitwidth);
    sampled_array_2 = sample_bool_array(bitwidth);
  
    // sampled_array_1 = vec![false, true, true, true, true, false, true, false];
    // sampled_array_2 = vec![false, true, true, true, false, true, false, false];
  
    println!("");
    println!("________");
    println!("FP ADDITION");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("sampled vector 2 {:?}", sampled_array_2);
  
    recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
    recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);
    sampled_array_3 = fp_add(&sampled_array_1, &sampled_array_2, expwidth, &special_encodings);
    recovered_value_3 = get_value(&sampled_array_3, expwidth, &special_encodings);
    
  
    println!("operation: fp_add");
    println!("returned vector: {:?}", sampled_array_3);
    println!("returned vector length: {}", sampled_array_3.len());
    reverted_vector = get_value(&sampled_array_3, expwidth, &special_encodings);
    println!("reverted vector: {}", reverted_vector);
    println!("");
    println!("{} + {} = {}", recovered_value_1, recovered_value_2, recovered_value_3);
    f32_comparison = recovered_value_1 + recovered_value_2;
    println!("f32 comparison: {}", f32_comparison);
    let recovered_array_from_f32 = encode_float(f32_comparison as f64, bitwidth, expwidth, &special_encodings);
    println!("comparison vector: {:?}", recovered_array_from_f32);
    println!("");

    
    
    if f32_comparison != reverted_vector && f32_comparison==f32_comparison && reverted_vector==reverted_vector {
      break
    }

    break
    
  }


  for i in 0..1 {
    
    // fp_add
    sampled_array_1 = sample_bool_array(bitwidth);
    sampled_array_2 = sample_bool_array(bitwidth);
  
    sampled_array_1 = vec![false, true, true, true, true, false, true, true];
    sampled_array_2 = vec![false, true, true, true, false, true, false, true];
  
    println!("");
    println!("________");
    println!("FP ADDITION (VARIABLE EXPONENT)");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("sampled vector 2 {:?}", sampled_array_2);
  
    recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
    recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);

    let mut output_expwidth:u8 = expwidth;
    
    (sampled_array_3, output_expwidth) = fp_add_variable_exp(&sampled_array_1, &sampled_array_2, expwidth, &special_encodings);

    
    recovered_value_3 = get_value(&sampled_array_3, output_expwidth, &special_encodings);
    
  
    println!("operation: fp_add_variable_exp");
    println!("returned vector: {:?}", sampled_array_3);
    println!("returned vector length: {}", sampled_array_3.len());
    reverted_vector = get_value(&sampled_array_3, output_expwidth, &special_encodings);
    println!("reverted vector: {}", reverted_vector);
    println!("");
    println!("{} + {} = {}", recovered_value_1, recovered_value_2, recovered_value_3);
    f32_comparison = recovered_value_1 + recovered_value_2;
    println!("f32 comparison: {}", f32_comparison);
    let recovered_array_from_f32 = encode_float(f32_comparison as f64, sampled_array_3.len() as u8, output_expwidth, &special_encodings);
    println!("comparison vector: {:?}", recovered_array_from_f32);
    println!("");

    
    
    if f32_comparison != reverted_vector && f32_comparison==f32_comparison && reverted_vector==reverted_vector {
      break
    }

    break
    
  }


  for i in 0..1 {
    
    // fp_add
    sampled_array_1 = sample_bool_array(bitwidth + 4);
    sampled_array_2 = sample_bool_array(bitwidth + 3);
  
    // sampled_array_1 = vec![false, true, true, true, true, false, true, true];
    // sampled_array_2 = vec![false, true, true, true, false, true, false, true];
  
    println!("");
    println!("________");
    println!("INTEGER ADDITION (EXTENDED)");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("sampled vector 2 {:?}", sampled_array_2);
  
    recovered_value_1 = integer_convert(&sampled_array_1) as f32;
    recovered_value_2 = integer_convert(&sampled_array_2) as f32;

    let mut output_expwidth:u8 = expwidth;
    
    sampled_array_3 = integer_add_extended(&sampled_array_1, &sampled_array_2, false, false, true);

// fn integer_add_extended(vector_one:&Vec<bool>, vector_two:&Vec<bool>, append_leading_bits:bool, assumed_leading_bit:bool, rightside_aligned:bool) -> Vec<bool> {

    
    recovered_value_3 = integer_convert(&sampled_array_3) as f32;
    
  
    println!("operation: integer_add_extended");
    println!("returned vector: {:?}", sampled_array_3);
    println!("returned vector length: {}", sampled_array_3.len());
    reverted_vector = integer_convert(&sampled_array_3) as f32;
    println!("reverted vector: {}", reverted_vector);
    println!("");
    println!("{} + {} = {}", recovered_value_1, recovered_value_2, recovered_value_3);
    f32_comparison = recovered_value_1 + recovered_value_2;
    println!("f32 comparison: {}", f32_comparison);
    let recovered_array_from_f32 = integer_revert(f32_comparison as u32, bitwidth);
    println!("comparison vector: {:?}", recovered_array_from_f32);
    println!("");

    
    
    if f32_comparison != reverted_vector && f32_comparison==f32_comparison && reverted_vector==reverted_vector {
      break
    }

    break
    
  }
  


  let mut f32_comparison : f32 = 0.;
  let mut recovered_array_from_f32 : Vec<bool> = vec![];

  for i in 0..1000 {
    
  
    // fp_subtract
    sampled_array_1 = sample_bool_array(bitwidth);
    sampled_array_2 = sample_bool_array(bitwidth);

    // sampled_array_1 = vec![];
    // sampled_array_2 = vec![];
    
    recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
    recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);

    println!("");
    println!("________");
    println!("FP SUBTRACTION");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("sampled vector 2 {:?}", sampled_array_2);
    println!("operation: fp_subtract");

    
    sampled_array_3 = fp_subtract(&sampled_array_1, &sampled_array_2, expwidth, &special_encodings);
    recovered_value_3 = get_value(&sampled_array_3, expwidth, &special_encodings);
  

    println!("returned vector: {:?}", sampled_array_3);
    println!("returned vector length: {}", sampled_array_3.len());
    reverted_vector = get_value(&sampled_array_3, expwidth, &special_encodings);
    println!("reverted vector: {}", reverted_vector);
    println!("");
    println!("{} - {} = {}", recovered_value_1, recovered_value_2, recovered_value_3);
    f32_comparison = recovered_value_1 - recovered_value_2;
    println!("f32 comparison: {}", f32_comparison);
    let recovered_array_from_f32 = encode_float(f32_comparison as f64, bitwidth, expwidth, &special_encodings);
    println!("comparison vector: {:?}", recovered_array_from_f32);
    println!("");

    
    
    if f32_comparison != reverted_vector && f32_comparison==f32_comparison && reverted_vector==reverted_vector {
      break
    }

    break
    
  }


  for i in 0..1000 {
    
  
    // fp_multiply
    sampled_array_1 = sample_bool_array(bitwidth);
    sampled_array_2 = sample_bool_array(bitwidth);

    // println!("UNDERFLOW SCENARIO");
    // sampled_array_1 = vec![true, false, false, false, true, false, false, true];
    // sampled_array_2 = vec![true, false, false, false, true, true, false, true];

    // println!("OVERFLOW SCENARIO");
    // sampled_array_1 = vec![true, true, true, false, false, true, false, false];
    // sampled_array_2 = vec![true, true, false, true, true, false, true, true];

    
    recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
    recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);

    println!("");
    println!("________");
    println!("FP MULTIPLY");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("sampled vector 2 {:?}", sampled_array_2);
    println!("operation: fp_multiply");

    
    sampled_array_3 = fp_multiply(&sampled_array_1, &sampled_array_2, expwidth, &special_encodings);
    recovered_value_3 = get_value(&sampled_array_3, expwidth, &special_encodings);
  

    println!("returned vector: {:?}", sampled_array_3);
    println!("returned vector length: {}", sampled_array_3.len());
    reverted_vector = get_value(&sampled_array_3, expwidth, &special_encodings);
    println!("reverted vector: {}", reverted_vector);
    println!("");
    println!("{} * {} = {}", recovered_value_1, recovered_value_2, recovered_value_3);
    f32_comparison = recovered_value_1 * recovered_value_2;
    println!("f32 comparison: {}", f32_comparison);
    let recovered_array_from_f32 = encode_float(f32_comparison as f64, bitwidth, expwidth, &special_encodings);
    println!("comparison vector: {:?}", recovered_array_from_f32);
    println!("");

    
    
    if f32_comparison != reverted_vector && f32_comparison==f32_comparison && reverted_vector==reverted_vector {
      break
    }

    // break
    
  }


  


  for i in 0..1000 {
    
  
    // fp_divide
    sampled_array_1 = sample_bool_array(bitwidth);
    sampled_array_2 = sample_bool_array(bitwidth);

    // sampled_array_1 = vec![true, false, true, true, true, false, false, true];
    // sampled_array_2 = vec![false, true, true, true, false, true, true, false];

    // println!("UNDERFLOW SCENARIO");
    // sampled_array_1 = vec![true, false, false, false, true, false, false, true];
    // sampled_array_2 = vec![true, false, false, false, true, true, false, true];

    // println!("OVERFLOW SCENARIO");
    // sampled_array_1 = vec![true, true, true, false, false, true, false, false];
    // sampled_array_2 = vec![true, true, false, true, true, false, true, true];

    
    recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
    recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);

    println!("");
    println!("________");
    println!("FP DIVIDE");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("sampled vector 2 {:?}", sampled_array_2);
    println!("operation: fp_divide");

    
    sampled_array_3 = fp_divide(&sampled_array_1, &sampled_array_2, expwidth, &special_encodings);
    // recovered_value_3 = get_value(&sampled_array_3, expwidth, &special_encodings);
    recovered_value_3 = get_value(&sampled_array_3, expwidth, &special_encodings);
  
    println!("");
    println!("returned vector: {:?}", sampled_array_3);
    println!("returned vector length: {}", sampled_array_3.len());
    reverted_vector = get_value(&sampled_array_3, expwidth, &special_encodings);
    println!("reverted vector: {}", reverted_vector);
    println!("");
    println!("{} / {} = {}", recovered_value_1, recovered_value_2, recovered_value_3);
    f32_comparison = recovered_value_1 / recovered_value_2;
    println!("f32 comparison: {}", f32_comparison);
    let recovered_array_from_f32 = encode_float(f32_comparison as f64, bitwidth * 2, expwidth, &special_encodings);
    println!("comparison vector: {:?}", recovered_array_from_f32);
    println!("");

    
    if f32_comparison != reverted_vector && f32_comparison==f32_comparison && reverted_vector==reverted_vector {
      break
    }

    break
    
  }





  // fp_compare

  let tolerance:Vec<bool> = vec![false, false, false, false, true, false, false, true, false];
  
  sampled_array_1 = sample_bool_array(bitwidth+4);
  sampled_array_2 = sample_bool_array(bitwidth);
  sampled_array_3 = sample_bool_array(bitwidth); // tolerance

  // sampled_array_1 = vec![false, false, false, true, true, false, true, false];
  // sampled_array_2 = vec![false, false, false, true, true, false, true, true];
  // sampled_array_3 = vec![false, false, false, false, true, false, true, true];; // tolerance

  println!("");
  println!("________");
  println!("FP COMPARE");
  println!("sampled vector 1 {:?}", sampled_array_1);
  println!("sampled vector 2 {:?}", sampled_array_2);
  println!("tolerance {:?}", sampled_array_3);
  println!("operation: fp_compare");

  
  recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
  recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);
  returned_bool = fp_compare(&sampled_array_1, &sampled_array_2, &tolerance, expwidth, expwidth, expwidth, &special_encodings);


  println!("returned bool: {}", returned_bool);
  println!("");
  println!("{} == {} = {}", recovered_value_1, recovered_value_2, returned_bool);
  println!("");
  // end fp compare validate
 


  // fp_match_precision
  for i in 0..1000 {
  
    let mut expwidth_1 = 2;
    let mut bitwidth_1 = bitwidth;
  
    sampled_array_1 = sample_bool_array(bitwidth_1);
    
    let mut expwidth_2 = 3;
    let mut bitwidth_2 = bitwidth + 2;
  
    sampled_array_2 = sample_bool_array(bitwidth_2);
  
  
    recovered_value_1 = get_value(&sampled_array_1, expwidth_1, &special_encodings);
    // recovered_value_2 = get_value(&sampled_array_2, expwidth_2, &special_encodings);
    // returned_bool = fp_compare(&sampled_array_1, &sampled_array_2, &tolerance, expwidth, expwidth, expwidth, &special_encodings);

    println!("");
    println!("________");
    println!("FP MATCH PRECISION");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("expwidth_1 = {}, bitwidth_1 = {}", expwidth_1, bitwidth_1);
    // println!("sampled vector 2 {:?}", sampled_array_2);
    println!("expwidth_2 = {}, bitwidth_2 = {}", expwidth_2, bitwidth_2);
    println!("operation: fp_match_precision");

  //  // fp_match_precision(vector_input:&Vec<bool>, expwidth_one:u8, bitwidth_two:u8, expwidth_two:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {

    sampled_array_3 = fp_match_precision(&sampled_array_1, expwidth_1, bitwidth_2, expwidth_2, &special_encodings);
    // recovered_value_3 = get_value(&sampled_array_3, expwidth, &special_encodings);
    recovered_value_3 = get_value(&sampled_array_3, expwidth_2, &special_encodings);

    println!("");
    println!("returned vector: {:?}", sampled_array_3);
    println!("returned vector length: {}", sampled_array_3.len());
    reverted_vector = get_value(&sampled_array_3, expwidth_2, &special_encodings);
    println!("reverted vector: {}", reverted_vector);
    println!("input value = {}", recovered_value_1);
    println!("output value = {}", reverted_vector);
    println!("");

    
    if recovered_value_1 != reverted_vector && recovered_value_1==recovered_value_1 && reverted_vector==reverted_vector {
      break
    }

    break
    
  }



  for i in 0..1000 {
  
    // fp_fma
    sampled_array_1 = sample_bool_array(bitwidth);
    sampled_array_2 = sample_bool_array(bitwidth);
    
    sampled_array_3 = sample_bool_array(bitwidth);
  
    let mut expwidth_3 = expwidth + 1;

    let mut f32_checkresult:f32;
  
  
    // sampled_array_1 = vec![true, false, true, true, true, true, true, false];
    // sampled_array_2 = vec![true, true, true, true, true, true, false, true];
    
    // sampled_array_3 = vec![false, false, false, false, true, false, false, true];
    
    
    recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
    recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);
    recovered_value_3 = get_value(&sampled_array_3, expwidth_3, &special_encodings);
  
    recovered_value_1 = get_value(&sampled_array_1, expwidth, &special_encodings);
    recovered_value_2 = get_value(&sampled_array_2, expwidth, &special_encodings);
    recovered_value_3 = get_value(&sampled_array_3, expwidth_3, &special_encodings);
  
    
    println!("");
    println!("________");
    println!("FP FMA");
    println!("");
    println!("fp_fma");
    println!("");
    println!("sampled vector 1 {:?}", sampled_array_1);
    println!("sampled vector 2 {:?}", sampled_array_2);
    println!("sampled vector 3 {:?}", sampled_array_3);
    println!("");
    println!("value 1 {:?}", recovered_value_1);
    println!("value 2 {:?}", recovered_value_2);
    println!("value 3 {:?}", recovered_value_3);
    println!("");

    f32_checkresult = (recovered_value_1 * recovered_value_2) + recovered_value_3;

    println!("f32: three += one * two = {}", f32_checkresult);
  
    (sampled_array_3, expwidth_3) = fp_fma(&sampled_array_1, &sampled_array_2, &sampled_array_3, expwidth, expwidth_3, &special_encodings);
  
    recovered_value_3 = get_value(&sampled_array_3, expwidth_3, &special_encodings);
  
    println!("");
    println!("returned sampled vector 3 {:?}", sampled_array_3);
    println!("returned expwidth_3 {}", expwidth_3);
    println!("");
    println!("value 3 {:?}", recovered_value_3);

    if f32_checkresult != recovered_value_3 && f32_checkresult==f32_checkresult && recovered_value_3==recovered_value_3 {
      break
    }

    break
    
  }


  
  for i in 0..1000 {
  
    // fp_dot
    sampled_array_1 = sample_bool_array(bitwidth * 3);
    sampled_array_2 = sample_bool_array(bitwidth * 3);
    
    sampled_array_3 = sample_bool_array(bitwidth);


    // sampled_array_1 = vec![false, false, false, false, false, true, true, false, false, true, false, true, true, false, true, false, false, true, true, false, true, true, false, false];
    // sampled_array_2 = vec![true, true, false, false, false, false, false, true, false, true, false, true, true, true, false, true, false, false, true, false, true, false, true, false];
    
    // sampled_array_3 = vec![true, false, false, false, false, true, true, false];
    

  
    let mut expwidth_3 = expwidth + 1;

    let mut f32_checkresult:f32;

    recovered_value_3 = get_value(&sampled_array_3, expwidth_3, &special_encodings);

    let mut sampled_entry_1_1:Vec<bool> = (sampled_array_1[0..(bitwidth as usize)]).to_vec();
    let mut sampled_entry_1_2:Vec<bool> = (sampled_array_1[(bitwidth as usize)..(2 * bitwidth as usize)]).to_vec();
    let mut sampled_entry_1_3:Vec<bool> = (sampled_array_1[(2 * bitwidth as usize)..(3 * bitwidth as usize)]).to_vec();

    let mut recovered_value_1_1 = get_value(&sampled_entry_1_1, expwidth, &special_encodings);
    let mut recovered_value_1_2 = get_value(&sampled_entry_1_2, expwidth, &special_encodings);
    let mut recovered_value_1_3 = get_value(&sampled_entry_1_3, expwidth, &special_encodings);

    let mut sampled_entry_2_1:Vec<bool> = (sampled_array_2[0..bitwidth as usize]).to_vec();
    let mut sampled_entry_2_2:Vec<bool> = (sampled_array_2[(bitwidth as usize)..(2 * bitwidth as usize)]).to_vec();
    let mut sampled_entry_2_3:Vec<bool> = (sampled_array_2[(2 * bitwidth as usize)..(3 * bitwidth as usize)]).to_vec();
    
    let mut recovered_value_2_1 = get_value(&sampled_entry_2_1, expwidth, &special_encodings);
    let mut recovered_value_2_2 = get_value(&sampled_entry_2_2, expwidth, &special_encodings);
    let mut recovered_value_2_3 = get_value(&sampled_entry_2_3, expwidth, &special_encodings);

    let mut dot_output:Vec<bool>;
    let mut dot_expwidth_output:u8;
    let mut f32_dot_checkvalue:f32;

    f32_dot_checkvalue = 
      recovered_value_3 +
      recovered_value_1_1 * recovered_value_2_1 +
      recovered_value_1_2 * recovered_value_2_2 +
      recovered_value_1_3 * recovered_value_2_3;

    (dot_output, dot_expwidth_output) = fp_dot(&sampled_array_1, &sampled_array_2, bitwidth, &sampled_array_3, expwidth, expwidth_3, &special_encodings);

    println!("");
    println!("________");
    println!("FP DOT");
    println!("");
    println!("fp_dot");
    println!("");
    println!("sampled_array_1 = {:?}", sampled_array_1);
    println!("sampled_array_2 = {:?}", sampled_array_2);
    println!("sampled_array_3 = {:?}", sampled_array_3);
    println!("");
    println!("{} + {} * {} + {} * {} + {} * {} = {}", 
      recovered_value_3,
      recovered_value_1_1,
      recovered_value_2_1,
      recovered_value_1_2,
      recovered_value_2_2,
      recovered_value_1_3,
      recovered_value_2_3,
      recovered_value_3 +
      recovered_value_1_1 * recovered_value_2_1 +
      recovered_value_1_2 * recovered_value_2_2 +
      recovered_value_1_3 * recovered_value_2_3
      );

    (dot_output, dot_expwidth_output) = fp_dot(&sampled_array_1, &sampled_array_2, bitwidth, &sampled_array_3, expwidth, expwidth_3, &special_encodings);

    let mut recovered_dot_output = get_value(&dot_output, dot_expwidth_output, &special_encodings);

    println!("recovered_dot_output = {}", recovered_dot_output);
    
    println!("f32_dot_checkvalue = {}", f32_dot_checkvalue);
    // println!("");
    // println!("value 1 {:?}", recovered_value_1);
    // println!("value 2 {:?}", recovered_value_2);
    // println!("value 3 {:?}", recovered_value_3);
    println!("");


    if f32_dot_checkvalue != recovered_dot_output && f32_dot_checkvalue==f32_dot_checkvalue && recovered_dot_output==recovered_dot_output {
      break
    }


    break
    
  }

  
}



// // ___________________________________________________________
// // *** [1] Configurable 754 Functions ***
// // encode_float(.) - derives an encoded vector of configurable 754 convention from a float
// // get_max_value(.) - returns max value for an encoding convention
// // get_value(.) - derives a float from an encoded vector of configurable 754 convention
// // get_magnitude(.) - support function to derive value of an exponent
// // get_mantissa(.) - support function to derive value of a mantissa

fn encode_float(target_number:f64, bitwidth:u8, expwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {

  // if value is reserved for a special encoding
  // return the special encoding
  let check_special_value_result:Vec<bool> = special_encodings.check_for_special_value(target_number as f32);
  if check_special_value_result != vec![] {
    return check_special_value_result;
  }

  let mut encoding:Vec<bool> = vec![false; bitwidth as usize];

  let mut number:f64 = target_number.clone();
  // println!("number = {}", number);

  // storing sign result in first entry of encoding
  // where (-1)^1 = -1 or (-1)^0 = 1
  
  if (number < 0.) || (number == 0. &&  f64::is_sign_negative(number)) {
    number = number.abs();
    encoding[0] = true;
  }
  // println!("number = {}", number);
  // println!("encoding = {:?}", encoding);

  // to access the exponent
  // will apply exponent = (log2(number) - min_exp).round()
  // where min_exp is derived as -1 * ( 2^(expwidth - 1) )
  // and the round operation rounds up
  // and the exponent is capped at 2^(expwidth - 1) - 1
  // let mut exp:f64 = number.clone();
  let max_exp:u32 = u32::pow(2, (expwidth - 1) as u32) - 1;
  // let min_exp:i32 = 1 - max_exp;

  // let mut exp:u32 = (number.log2() - (1. - max_exp as f64)).ceil() as u32; // this version using ceil() rounding has edge case for max mantissa so better to use floor() rounding
  let mut exp:u32 = (number.log2() + (max_exp as f64)).floor() as u32;

  // note that the encoded exp is not to be confused
  // with the applied exponent to convert back to float
  // which would be applied as 2^(exp - exp_max)  
  
  // println!("max_exp = {}", max_exp);
  // println!("applied exp = {}", exp - max_exp);
  // println!("exp = {}", exp);



  // great now we can encode the mantissa

  // we'll convert the number variable to the mantissa value
  number /= f64::powf(2., (exp as i32 - max_exp as i32) as f64);
  // println!("number = {}", number);
  number -= 1.;

  // we'll initialize variable to derive the maximum encodable mantissa value
  // for informational purposes
  let mut max_mantissa:f64 = 1.;
  let mantissa_width:u8 = bitwidth - 1 - expwidth;

  // now in parallel we'll calc max mantissa and populate encoding
  for i in 0..mantissa_width {

    let power:f64 = f64::powf(2., -1. * (i as f64 + 1.));

    if number >= power {
      let encoding_address:usize = i as usize + 1 + expwidth as usize;
      encoding[encoding_address] = true;
      number -= power;
    }
    
    max_mantissa += power;
  }
  // println!("max_mantissa = {}", max_mantissa);

  // ok now can encode the exponent
  // let encoding_stop:u8 = 1 + expwidth;
  
  // for (i, encoding_index) in (1 ..encoding_stop as u8).enumerate() {

  //   let two_exponent:u32 = u32::pow(2, (expwidth - i as u8) as u32);
  //   println!("two_exponent = {}", two_exponent);
  //   println!("exp = {}", exp);
    
  //   if exp as u32 >= two_exponent {
  //     encoding[encoding_index as usize] = 1u8;
  //     exp -= two_exponent;
  //   }
  // }

  let encoding_stop:u8 = 1 + expwidth;
  
  for (i, encoding_index) in (1 ..encoding_stop as u8).enumerate() {

    // println!("i = {}", i);
    // println!("encoding_index = {}", encoding_index);
    // println!("expwidth - i = {}", expwidth - i as u8);

    let two_exponent:u32 = u32::pow(2, (expwidth - i as u8 - 1) as u32);
    // println!("two_exponent = {}", two_exponent);
    // println!("exp = {}", exp);
    
    if exp as u32 >= two_exponent {
      encoding[encoding_index as usize] = true;
      exp -= two_exponent;
    }
  }
  
  return encoding;  
}

fn get_max_value(bitwidth:u8, expwidth:u8, special_encodings:&SpecialEncodings) -> f32 {
  // returns the max value associated with configurable 754 convention
  // basically max encoding is associated with 0 exp and max mantissa

  let mut max_vector:Vec<bool> = vec![true; bitwidth as usize];

  //sign will be zero for max value
  max_vector[0] = false;

  let max_value = get_value(&max_vector, expwidth, &special_encodings);

  return max_value;
}

fn get_value(sampled_array : &Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> f32 {
  // this derives a value for an encoded vector
  // follows basic 754 conventions with configurable bitwidth and exponent width
  // note that currently implemented without special values as a starting point

  // if vector is reserved for a special value
  // return the special value
  let check_special_value_result:f32 = special_encodings.check_for_reserved_encoding(&sampled_array);
  if check_special_value_result != 1.0f32 {
    return check_special_value_result;
  }

  let bitwidth:u8 = sampled_array.len() as u8;

  //derived values
  let max_exp:i8 = (u32::pow(2, expwidth as u32 -1) - 1) as i8;
  let min_exp:i8 = 1 - max_exp;

  // println!("bitwidth = {}", bitwidth);
  // println!("expwidth = {}", expwidth);
  // println!("max_exp = {}", max_exp);
  // println!("min_exp = {}", min_exp);

  // this calculates value of sign
  let sign:i16 = i32::pow(-1, sampled_array[0] as u32) as i16;
  // println!("sign = {}", sign);

  // this calculates value of exponent
  let mut start:u8 = 1;
  let mut stop:u8 = 1+expwidth;
  let exp:i16 = get_magnitude(&sampled_array, start, stop) as i16 - max_exp as i16;
  // println!("exp = {}", exp);

  // this calculates magnitude of mantissa
  start = 1+expwidth;
  stop = bitwidth;
  let mantissa : f32 = get_mantissa(&sampled_array, start, stop) as f32;
  // println!("mantissa = {}", mantissa);

  let value : f32 = sign as f32 * mantissa as f32 * (f32::powf(2., exp as f32) as f32);
  // println!("value = {}", value);

  return value;
  
}

fn get_magnitude(sampled_array : &Vec<bool>, start:u8, stop:u8) -> i16 {
  // this calculates magnitude of exponent
  // used as support function in get_value

  let mut magnitude:i16 = 0;

  let bit_register_count: u8 = stop - start;

  // in an alternate configuration 
  // this exponents_vector could be a global constant initialized externally
  // which would make this more efficient
  // this approach is for flexibility associated with variable bit allocations
  let mut exponents_vector = vec![0i16; bit_register_count as usize];
  for _entry in 0..bit_register_count { //0..bitwidth ~ range(bitwidth)
    exponents_vector[_entry as usize] = i32::pow(2, (bit_register_count - _entry - 1) as u32) as i16;
    // exponents_vector[_entry as usize] = i32::pow(2, _entry as u32) as i16;
  }

  // println!("exponents_vector = {:?}", exponents_vector);

  let magnitude_slice = &sampled_array[start as usize ..stop as usize];
  // println!("magnitude_slice = {:?}", magnitude_slice);

  for _entry in 0..bit_register_count { //0..bitwidth ~ range(bitwidth)
    // magnitude = magnitude + (exponents_vector[_entry as usize] as i16 * magnitude_slice[_entry as usize] as i16) as i16;
    magnitude += (exponents_vector[_entry as usize] as i16 * magnitude_slice[_entry as usize] as i16) as i16;
  }

  // println!("derived exponent = {}", magnitude);
  
  return magnitude;
}


fn get_mantissa(sampled_array : &Vec<bool>, start:u8, stop:u8) -> f32 {
  // this calculates magnitude of mantissa
  // used as a support function in both get_value() and get_posits_value()

  let mut magnitude:f32 = 0.;

  let bit_register_count: u8 = stop - start;

  // in an alternate configuration 
  // this mantissa exponents_vector could be a global constant initialized externally
  // which would make this more efficient
  // this approach is for flexibility associated with variable bit allocations
  let mut exponents_vector = vec![0.; bit_register_count as usize];
  for _entry in 0..bit_register_count { //0..bitwidth ~ range(bitwidth)
    exponents_vector[_entry as usize] = f32::powf(2., -1. as f32 - _entry as f32) as f32;
  }

  // println!("mantissa exponents_vector = {:?}", exponents_vector);

  let magnitude_slice = &sampled_array[start as usize ..stop as usize];
  // println!("mantissa magnitude_slice = {:?}", magnitude_slice);

  for _entry in 0..bit_register_count { //0..bitwidth ~ range(bitwidth)
    magnitude = magnitude + (exponents_vector[_entry as usize] * magnitude_slice[_entry as usize] as u32 as f32) as f32;
  }

  magnitude = magnitude + 1.;
  
  return magnitude;
}

// // ___________________________________________________________
// // *** [2] Posits Functions ***
// // ### note that posits functions aren't yet passing validations,
// // ### for now consider this an outline pending further QC
// // get_max_posits_value(.) - derives a max value available to a posits configuration
// // get_posits_value(.) - derives a float from an encoded vector of posits convention
// // encode_posits_float(.) - derives a bool vector encoding from a float
// // populate_posit_config(.) - populates a PositSupport struct used in encode_posits_float

fn encode_posits_float(target_number:f64, bitwidth:u8, expwidth:u8, posit_config:&PositSupport) -> Vec<bool> {
  // derives an encoded vector of posits convention from a float
  // fow now will return encoding with same number of registers
  // (need to put some thought into rounding conventions)
  // however in posits there may be gaps between distribution buckets
  // in which case populates a vector associated with nearest min entry and either min or max mantissa
  // just to pick a convention

  // println!("one");

  let mut output_vector:Vec<bool> = vec![];

  let max_abs_set:&Vec<f64> = &posit_config.max_abs_set;
  let min_abs_set:&Vec<f64> = &posit_config.min_abs_set;

  let mut index_max_abs:usize = 0;
  let mut index_min_abs:usize = 0;

  // find index of nearest posits distribution bucket
  for (i, entry) in max_abs_set.iter().enumerate() {
    if &target_number.abs() > entry {
      index_max_abs = i;
      break
    }
  }

  for (j, entry) in min_abs_set.iter().enumerate() {
    if &target_number.abs() > entry {
      index_min_abs = j;
      break
    }
  }

  // println!("two");
  
  let mut basis_index:usize = 0;

  // if falls within a min/max bucket keep index
  // else if outside of a bucket find nearest min index
  // (this is a form of rounding)
  if index_max_abs > index_min_abs {
    // this is case where value falls within a min / max bucket
    // so will use index_min_abs as basis
    basis_index = index_min_abs;
  } else {
    // this is case where value falls outside a bucket
    // so find nearest bucket
    if index_min_abs > 0 {
      if (&target_number.abs() - min_abs_set[index_min_abs]).abs() <= (&target_number.abs() - min_abs_set[index_min_abs - 1]).abs() {
        basis_index = index_min_abs;
      } else {
        basis_index = index_min_abs - 1;
      }
    }
  }

  // now that we know the basis index, just a matter of deriving the mantissa
  // println!("three");

  let sign_bit:bool = f64::is_sign_negative(target_number);
  let regime_bit:bool = posit_config.regime_set[basis_index] != 0;
  let cons_count:u32 = posit_config.cons_set[basis_index];
  let mut end_of_exp_bit:Vec<bool> = vec![]; // left empty when full expwdith is applied
  if basis_index != 0 && basis_index != ((2 * expwidth) - 1) as usize {
    end_of_exp_bit.push(!regime_bit);
  }
  let exp_bits_vec:Vec<bool> = vec![regime_bit; cons_count as usize];

  // let's populate everything except the mantissa in output_vector
  output_vector.push(sign_bit);
  for entry in exp_bits_vec {
    output_vector.push(entry);
  }
  for entry in end_of_exp_bit {
    output_vector.push(entry);
  }

  // mantissa width is whatever is left
  let mantissa_width:u8 = bitwidth - output_vector.len() as u8;

  // deriving a mantissa depends on whether falls within a distribution bucket
  // we'll divide by min of the bucket and set floor / cap based on mantissa capacity

  // troubleshoot
  // println!("troubleshoot");
  // println!("&target_number.abs() = {}", &target_number.abs());
  // println!("min_abs_set[basis_index] = {}", min_abs_set[basis_index]);
  // println!("&target_number.abs() / min_abs_set[basis_index] = {}", &target_number.abs() / min_abs_set[basis_index]);
  
  let mut mantissa_value:f64 = &target_number.abs() / min_abs_set[basis_index];

  // if mantissa_value < 1. {
  //   mantissa_value = 1.;
  // } else if mantissa_value > max_abs_set[basis_index] {
  //   mantissa_value = max_abs_set[basis_index];
  // }

  // println!("mantissa_value = {}", mantissa_value);

  // now we can extract the mantissa bits in the usual way
  let mut mantissa_bits:Vec<bool> = vec![];

  mantissa_value -= 1.;
  for i in 1..(mantissa_width+1) {

    let power:f64 = f64::powf(2., -(i as f64));
    
    if mantissa_value >= power {
      mantissa_bits.push(true);
      mantissa_value -= power;
    } else {
      mantissa_bits.push(false);
    }
  }

  // great now merge mantissa bits onto output vector and good to go
  for entry in mantissa_bits {
    output_vector.push(entry);
  }
  
  return output_vector;
}

fn populate_posit_config(bitwidth:u8, expwidth:u8, k_multiplier:f64) -> PositSupport {
  // populates a data structure used to translate floats to encoding
  // k_convention expects one of {"default", "alternate"}

  // here is the full struct convention
  //   struct PositSupport {
  //   expwidth : u8,
  //   k_multiplier : f64, // traditionally defaults to 1.0, for smaller bitwidths may be appropriate to scale down e.g. 0.33 or something
  //   k_set : Vec<f64>,
  //   u_to_k_set : Vec<f64>,
  //   cons_set : Vec<u32>,
  //   two_to_cons_set : Vec<u32>,
  //   u_to_k_times_two_to_cons_set : Vec<f64>,
  //   max_mantissa_set : Vec<f64>,
  //   max_abs_set : Vec<f64>,
  //   min_abs_set : Vec<f64>,
  // }

  // println!("expwidth = {}", expwidth);
  // println!("k_multiplier = {}", k_multiplier);

  // this might be a little hard to follow along
  // I have a spreadsheet that spells it out offline
  // basically using the convention to populate these vectors
  // of starting in 1 regime (where exponent is populated with 1's)
  // and incrementing down number of consecutive exponent bits
  // then doing the 0 regime (where exponent populated with zero's)
  // starting at 1 exponent bit and incrementing back up
  // to maximum number of exponent bits
  // this order results in populating a vector of min's or max's for distribution buckets from largest to smallest
  // which will allow matching a target float to a specific bucket and then deriving a resulting mantissa value
  // which should be much cheaper than a full lookup table

  // traditionally k_multiplier defaults to 1.0, 
  // for smaller bitwidths I find it may be appropriate to scale down 
  // e.g. for 8 bit can use 0.33 or something
  // which makes the gaps between distribution buckets a little less jagged
  let mut k_set:Vec<f64> = vec![];
  for i in 0..expwidth {
    // start with the 1 regime basis k values
    k_set.push( ((expwidth - i) as f64 -1.) * k_multiplier);
  }
  for i in 1..(expwidth+1) {
    // now apply the 0 regime basis k values
    k_set.push( (-1. * i as f64) * k_multiplier);
  }

  // println!("k_set = {:?}", k_set);

  let u:f64 = f64::powf(2., f64::powf(2., expwidth as f64));

  // println!("u = {}", u);

  let mut u_to_k_set:Vec<f64> = vec![];
  for entry in k_set.iter() {
    u_to_k_set.push( f64::powf(u, *entry as f64 ) );
  }

  // println!("u_to_k_set = {:?}", u_to_k_set);
  
  let mut cons_set:Vec<u32> = vec![];
  // these values correspond to basis used to populate k_set, etc
  for i in 0..(expwidth) {
    cons_set.push( (expwidth - i) as u32 );
  }
  for i in 1..(expwidth+1) {
    cons_set.push( i as u32 );
  }

  // println!("cons_set = {:?}", cons_set);

  let mut regime_set:Vec<u8> = vec![];
  for _i in 0..expwidth {
    regime_set.push(1);
  }
  for _i in 0..expwidth {
    regime_set.push(0);
  }

  // println!("regime_set = {:?}", regime_set);

  let mut two_to_cons_set:Vec<u32> = vec![];
  for entry in cons_set.iter() {
    two_to_cons_set.push( u32::pow(2, *entry));
  }

  // println!("two_to_cons_set = {:?}", two_to_cons_set);

  let mut u_to_k_times_two_to_cons_set:Vec<f64> = vec![];
  for (i, entry) in u_to_k_set.iter().enumerate() {
    u_to_k_times_two_to_cons_set.push( *entry * two_to_cons_set[i] as f64 );
  }

  // println!("u_to_k_times_two_to_cons_set = {:?}", u_to_k_times_two_to_cons_set);

  let mut max_mantissa_set:Vec<f64> = vec![];
  let mut max_mantissa:f64;
  let mut end_of_exp_bit:u8 = 1;
  
  for i in 0..expwidth {
    
    if i == 0 {
      end_of_exp_bit = 0;
    } else {
      end_of_exp_bit = 1;
    }

    max_mantissa = 1.;
    
    // start with the 1 regime basis k values
    let mantissa_width = bitwidth as u8 - 1 - (expwidth - i) as u8 - end_of_exp_bit;

    // println!("mantissa_width = {}", mantissa_width);

    for i in 1..(mantissa_width+1) {
      max_mantissa += f64::powf(2., -(i as f64));
    }
    // println!("max_mantissa = {}", max_mantissa);

    max_mantissa_set.push(max_mantissa);
  }
  for i in 1..(expwidth+1) {

    if i == (expwidth) {
      end_of_exp_bit = 0;
    } else {
      end_of_exp_bit = 1;
    }

    max_mantissa = 1.;
    
    // now apply the 0 regime basis k values
    let mantissa_width = bitwidth as u8 - 1 - end_of_exp_bit - i as u8;

    // println!("mantissa_width = {}", mantissa_width);

    for i in 1..(mantissa_width+1) {
      max_mantissa += f64::powf(2., -(i as f64));
    }

    // println!("max_mantissa = {}", max_mantissa);

    max_mantissa_set.push(max_mantissa);
  }

  // println!("max_mantissa_set = {:?}", max_mantissa_set);

  let mut max_abs_set:Vec<f64> = vec![];
  let mut min_abs_set:Vec<f64> = vec![];

  for (i, entry) in u_to_k_times_two_to_cons_set.iter().enumerate() {

    max_abs_set.push(*entry * max_mantissa_set[i]);
    min_abs_set.push(*entry * 1.);
    
  }
  
  
  // println!("max_abs_set = {:?}", max_abs_set);
  // println!("min_abs_set = {:?}", min_abs_set);

  let posit_config = PositSupport {
    expwidth : expwidth,
    k_multiplier : k_multiplier,
    k_set : k_set,
    u : u,
    u_to_k_set : u_to_k_set,
    cons_set : cons_set,
    regime_set : regime_set,
    two_to_cons_set : two_to_cons_set,
    u_to_k_times_two_to_cons_set : u_to_k_times_two_to_cons_set,
    max_mantissa_set : max_mantissa_set,
    max_abs_set : max_abs_set,
    min_abs_set : min_abs_set,
  };

  return posit_config;
  
}

fn get_max_posits_value(bitwidth:u8, expwidth:u8) -> f32 {
  // returns the max value associated with configurable 754 convention
  // basically max encoding is associated with 0 exp and max mantissa

  let mut max_vector:Vec<bool> = vec![true; bitwidth as usize];

  //sign will be zero for max value
  max_vector[0] = false;

  let max_value = get_posits_value(&max_vector, bitwidth, expwidth);

  // println!("max_value per 1's exponent {}", max_value);

  // let mut max_vector2:Vec<u8> = vec![0; bitwidth as usize];
  // let max_value2 = get_posits_value(&max_vector2, bitwidth, expwidth);
  // println!("max_value per 0's exponent {}", max_value2);

  return max_value;
}

fn get_posits_value(sampled_array : &Vec<bool>, bitwidth:u8, expwidth:u8) -> f32 {
  // this derives a posits value for a sampled vector
  // for documentation on terminology and methodology
  // recommend https://www.johndcook.com/blog/2018/04/11/anatomy-of-a-posit-number/

  // this calculates value of sign
  let sign:i16 = i32::pow(-1, sampled_array[0] as u32) as i16;
  // println!("posits sign = {}", sign);

  // determine regime basis of 0 or 1
  let regime_basis:bool = sampled_array[1];
  // println!("regime_basis = {}", regime_basis);

  // determine number of consecutive regime basis values
  let mut consecutive_regime_bit_count:u8 = 1;
  for i in 2..(expwidth+3) {
    // println!("i = {}", i);
    // println!("sampled_array[i as usize] as u8 != regime_basis {}", sampled_array[i as usize] != regime_basis);

    if i == expwidth+2 {
      consecutive_regime_bit_count += (i-3) as u8;
      break
    } else if sampled_array[i as usize] != regime_basis {
      consecutive_regime_bit_count += (i-2) as u8;
      break
    }
  }
  // println!("consecutive_regime_bit_count = {}", consecutive_regime_bit_count);

  
  let exp_stop:u8 = consecutive_regime_bit_count + 1;

  let mut fraction_start:u8 = exp_stop + 1;
  if consecutive_regime_bit_count == expwidth {
    fraction_start -= 1;
  }
  // println!("exp_stop = {}", exp_stop);
  // println!("fraction_start = {}", fraction_start);

  // the k_exponent is a posits thing based on regime_basis (of 0's or 1's)
  let k_exponent:i8 = if regime_basis == false {
    // for the zero regime k_exponent = - consecutive_regime_bit_count
    let k_exponent:i8 = -1 * consecutive_regime_bit_count as i8;
    k_exponent
  } else {
    // for the one regime k_exponent = consecutive_regime_bit_count - 1
    let k_exponent:i8 = consecutive_regime_bit_count as i8 - 1;
    k_exponent
  };
  // println!("k_exponent = {}", k_exponent);

  // useed is another posits convention, interpretted as 2^(2^expwidth)
  // (based on maximum expwidth)
  let useed:u32 = u32::pow(2, u32::pow(2, expwidth as u32));
  // println!("useed = {}", useed);

  // whatever bits are left after the exponent terminating bit belong to fraction
  let fraction_bit_count:u8 = if bitwidth > consecutive_regime_bit_count + 2 {
    if consecutive_regime_bit_count < expwidth {
      let fraction_bit_count:u8 = bitwidth - consecutive_regime_bit_count - 2;
      fraction_bit_count
    } else {
      let fraction_bit_count:u8 = bitwidth - consecutive_regime_bit_count - 1;
      fraction_bit_count
    }

  } else {
    0u8
  };
  // println!("fraction_bit_count = {}", fraction_bit_count);

  // now we can dervie our fraction values in a manner resembling how derive a mantissa in 754
  let fraction:f32 = if fraction_start < bitwidth {
    let fraction:f32 = get_mantissa(&sampled_array, fraction_start, bitwidth) as f32;
    fraction
  } else {
    1.
  };
  // println!("fraction = {}", fraction);

  
  // putting it all together:
  let value:f32 = sign as f32 * f32::powf(useed as f32, k_exponent as f32) as f32 * f32::powf(2., consecutive_regime_bit_count as f32) as f32 * fraction as f32;

  // println!("value = {}", value);

  return value;
}

// ___________________________________________________________
// *** [3] Methods to Sample Encodings ***
// sample_array(.) - randomly samples a bitwidth length vector of u8 1/0 entries
// sample_bool(.) - weighted sampling of a single bool
// sample_bool_array(.) - randomly samples a bitwidth length vector of bool entries
// get_all_encodings(.) - generates set of all encodings associated with a bit width
// get_curent_encoding(.) - support function to generate a specific integer encoding
// get_all_values(.) - translate array of bits to array of values per configurable 754
// validate_all_values(.) - encodes and reverts each value to validate
// get_saturation_and_zero_encodings(.) - this returns a tuple in form: (max_value, max_encoding, min_value, min_encoding, zero_value, zero_encoding, negzero_value, negzero_encoding)


fn sample_array(bitwidth:u8) -> Vec<u8> { //vec![u8; BITWIDTH as usize] {
  // basically this randomly samples a bitwidth length vector of u8 1/0 entries
  // intended as a resource to generate random encodings

  // println!("{}", bitwidth);
  let mut rng = thread_rng(); // initialize random number generator
  let mut sampled_array:Vec<u8> = vec![]; // initialize array with zeros
  
  for _entry in 0..bitwidth { //0..bitwidth ~ range(bitwidth)
    let a:u8 = rng.gen_range(0..=1); //generates a random bit
    // println!("{}", a); //inspect
    sampled_array.push(a);
  }

  // println!("the new array is {:?}", sampled_array);
  return sampled_array;
}

fn sample_bool(weight:Vec<bool>) -> bool {
  // sample a single boolean
  // weights the operation by deriving a float between 0-1
  // based on a recieved bool vector
  // with assumed registers of 2^-1, 2^-2, 2^-3, etc.
  // in the interest of avoid an imbalance from 0 weight for all null and a <1 weight for all true
  // will have a small offset to the fraction by adding a trailing true register to the received vector
  // which results in the delta (min weight - 0) = (1 - max weight)

  // for cases where the weight parameter recieved as emntpy vector vec![]
  // will apply a default 50% weigth

  // obviously a future extension might incorporate parameters to integrate variations on e.g. pseudo random number generator and etc
  // the goal is to keep such aspects modular

  // we'll use a lookup table to keep latency down
  // (referring to "multiplier_lookup" vector below)

  let mut weight_float:f32;
  let weight_len:usize = weight.len();

  if weight_len == 0 {
    weight_float = 0.5;
    
  } else {

    // weight.push(true);
    weight_float = 0.;

    let multiplier_lookup:Vec<f32> = vec![0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];

    let mut iteration_count:usize = 0;
    
    for entry in weight.iter() {

      if *entry {
        weight_float += multiplier_lookup[iteration_count];
      }

      iteration_count += 1;

      // to keep this manageable limit to first 6 passed bits
      if iteration_count==6 {
        break
      }
    }

    // this is the trailing true register noted in comments
    weight_float += multiplier_lookup[iteration_count];

  }
  
  let mut rng = thread_rng(); // initialize random number generator
  let a_sample: bool = thread_rng().gen_bool(weight_float as f64); // sample at 50%
  
  return a_sample;
}

fn sample_bool_array(bitwidth:u8) -> Vec<bool> { //vec![u8; BITWIDTH as usize] {
  // basically this randomly samples a bitwidth length vector of u8 1/0 entries
  // intended as a resource to generate random encodings

  // println!("{}", bitwidth);
  let mut rng = thread_rng(); // initialize random number generator
  let mut sampled_array:Vec<bool> = vec![]; // initialize array with zeros
  
  for _entry in 0..bitwidth { //0..bitwidth ~ range(bitwidth)

    let a_sample: bool = thread_rng().gen_bool(0.5); // sample at 50%
    
    sampled_array.push(a_sample);
  }

  // println!("the new array is {:?}", sampled_array);
  return sampled_array;
}


fn get_all_encodings(bitwidth:u8) -> Vec<bool> {
  // rather than sample random encodings, 
  // it might also be useful to access the set of all encodings
  // will return as a vector of length [bitwidth * 2^bitwidth]

  //rust isn't great about populating vectors of vectors
  //so instead will just have one vector 
  //of length bitwidth * encoding_count
  
  let encoding_count:u32 = u32::pow(2, bitwidth as u32);
  // println!("encoding_count {}", encoding_count);

  let vector_length:u32 = encoding_count * bitwidth as u32;
  // println!("vector_length {}", vector_length);

  let mut all_encodings:Vec<bool> = vec![false; vector_length as usize];

  let mut start:usize = 0;
  let mut stop:usize = bitwidth as usize;
  
  for i in 0..encoding_count {

    let current_encoding:Vec<bool> = get_curent_encoding(i, bitwidth);

    for (index_current, index_all) in (start as u32 ..stop as u32).enumerate() {
      all_encodings[index_all as usize] = current_encoding[index_current as usize];
    }
    
    start += bitwidth as u32 as usize;
    stop += bitwidth as usize;

    // println!("i = {}", i);
    // println!("start = {}", start);
    // println!("stop = {}", stop);
    // println!("current_encoding = {:?}", current_encoding);
    
  }
  
  return all_encodings;
}


fn get_curent_encoding(i:u32, bitwidth:u8) -> Vec<bool> {
  // this is a support function for get_all_encodings
  // i will be integer between 0..encoding_count
  // this returns a vector of length bitwidth
  // e.g. for i=13 and bitwidth=8, returns
  // vec![0,0,0,0,1,1,0,1]
  // (basically returns an integer encoding)

  let mut j:u32 = i;

  let mut current_encoding:Vec<bool> = vec![false; bitwidth as usize];

  let mut exponent:u32;

  // let mut exponents:Vec<u32> = vec![0u32; bitwidth as usize];

  for k in 0..bitwidth {
    exponent = u32::pow(2, (bitwidth - k - 1) as u32);

    if j >= exponent {
      current_encoding[k as usize] = true;
      j -= exponent;
    }
    
  }

  // println!("{:?}", current_encoding);

  return current_encoding;
}

fn get_all_values(sampled_array : &Vec<bool>, bitwidth:u8, expwidth:u8, special_encodings:&SpecialEncodings, convention:&str) -> Vec<f32> {
  // function to translate a populated vector of all encodings
  // to a vector of all corresponding values
  // convention accepts &str as one of {"754", "posits"}
  
  let mut values:Vec<f32> = vec![];

  // if convention is "754" then use the configurable 754 convention

  let total_encodings:u32 = u32::pow(2, bitwidth as u32);

  let mut start:usize = 0;
  let mut stop:usize = bitwidth as usize;

  for i in 0..total_encodings {

    let slice:Vec<bool> = (&sampled_array[start..stop]).to_vec();

    let value:f32 = if convention == "754" {
      let value:f32 = get_value(&slice, expwidth, &special_encodings);
      value
    // } else if convention == "posits" {
    } else {
      let value:f32 = get_posits_value(&slice, bitwidth, expwidth);
      value
      // 1. //plug value
    };
    
    values.push(value);

    start += bitwidth as usize;
    stop += bitwidth as usize;
    
  }

  return values;
}

fn validate_all_values(all_values:&Vec<f32>, bitwidth:u8, expwidth:u8, special_encodings:&SpecialEncodings) -> bool {
  // having converted all encodings to values
  // we can then convert all values back to encodings and revert
  // to validate full consistency in both directions

  let mut result:bool = false;
  let mut mistmatch_vector:Vec<f32> = vec![];
  let mut recovered_mismatch_encoding:Vec<bool> = vec![];
  let mut recovered_mismatch_encoding_vector:Vec<bool> = vec![];  
  
  for entry_value in all_values.iter() {

    // _____scenario A_____
    // in this version only encode revert once
    
    let entry_encoding:Vec<bool> = encode_float(entry_value.clone() as f64, bitwidth, expwidth, &special_encodings);

    let reverted_value:f32 = get_value(&entry_encoding, expwidth, &special_encodings);
    // // _____end scenario A_____

    // // _____scenario B_____
    // // just to be thorough can alternatively encode revert twice
    
    // let entry_encoding_0:Vec<bool> = encode_float(entry_value.clone() as f64, bitwidth, expwidth, &special_encodings);

    // let reverted_value_0:f32 = get_value(&entry_encoding_0, expwidth, &special_encodings);

    // let entry_encoding:Vec<bool> = encode_float(reverted_value_0.clone() as f64, bitwidth, expwidth, &special_encodings);

    // let reverted_value:f32 = get_value(&entry_encoding, expwidth, &special_encodings);
    // // _____end scenario B_____
    

    if entry_value.clone() as f32 != reverted_value {

      // println!("");
      // println!("mistmactch:");
      // println!("entry_value = {}", entry_value);
      // println!("reverted_value = {}", reverted_value);
      // println!("entry_encoding = {:?}", entry_encoding);

      result = true;

      mistmatch_vector.push(entry_value.clone() as f32);
      mistmatch_vector.push(reverted_value);

      recovered_mismatch_encoding = encode_float(reverted_value.clone() as f64, bitwidth, expwidth, &special_encodings);

      for entry2 in entry_encoding.iter() {
        recovered_mismatch_encoding_vector.push(*entry2);
      }
      for entry2 in recovered_mismatch_encoding.iter() {
        recovered_mismatch_encoding_vector.push(*entry2);
      }
    }
  }
  println!("");
  println!("test complete");

  println!("{:?}", mistmatch_vector);

  let mistmatch_vector_length:u16 = mistmatch_vector.len() as u16;

  let mut start = 0 as usize;
  let mut stop = bitwidth as usize;

  println!("mistmatch_vector_length {}", mistmatch_vector_length);
  println!("recovered_mismatch_encoding_vector length {}", recovered_mismatch_encoding_vector.len());
  println!("");
  
  for i in 0..mistmatch_vector_length {
    
    let encoding_slice1 = &recovered_mismatch_encoding_vector[start..stop];

    println!("{}", mistmatch_vector[i as usize] as f32);
    println!("{:?}", encoding_slice1);

    start += bitwidth as usize;
    stop += bitwidth as usize;

    if i%2 > 0 {
      println!("");
    }
    
  }
  
  return result;
}

fn get_saturation_and_zero_encodings(bitwidth:u8, expwidth:u8, special_encodings:&SpecialEncodings, convention:&str) -> (f32, Vec<bool>, f32, Vec<bool>, f32, Vec<bool>, f32, Vec<bool>) {
  // for a given configuration
  // returns a tuple of saturation values / encodings and +/- zero values / encodings

  // this showsall encodings for a given bit width
  let all_encodings:Vec<bool> = get_all_encodings(bitwidth);

  // this returns all values associated with the convention
  let all_values:Vec<f32> = get_all_values(&all_encodings, bitwidth, expwidth, &special_encodings, convention);

  let mut max_value:f32 = 0.;
  let mut max_encoding:Vec<bool> = vec![];
  let mut min_value:f32 = 0.;
  let mut min_encoding:Vec<bool> = vec![];
  let mut zero_value:f32 = 0.;
  let mut zero_encoding:Vec<bool> = vec![];
  let mut negzero_value:f32 = -0.;
  let mut negzero_encoding:Vec<bool> = vec![];

  //calculates the min float in a vector
  let mut min_value = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
  println!("min_value = {}", min_value);

  //calculates the max float in a vector
  let mut max_value = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
  println!("max_value = {}", max_value);

  if convention == "754" {
    max_encoding = encode_float(max_value as f64, bitwidth, expwidth, &special_encodings);
    min_encoding = encode_float(min_value as f64, bitwidth, expwidth, &special_encodings);
    zero_encoding = encode_float(zero_value as f64, bitwidth, expwidth, &special_encodings);
    negzero_encoding = encode_float(negzero_value as f64, bitwidth, expwidth, &special_encodings);
  } else {

    // not yet defined for posits
    
    // max_encoding = encode_posits_float(max_value as f64, bitwidth, expwidth);
    // min_encoding = encode_posits_float(min_value as f64, bitwidth, expwidth);
    // zero_encoding = encode_posits_float(zero_value as f64, bitwidth, expwidth);
    // negzero_encoding = encode_posits_float(negzero_value as f64, bitwidth, expwidth);
  }

  let return_tuple : (f32, Vec<bool>, f32, Vec<bool>, f32, Vec<bool>, f32, Vec<bool>) = (max_value, max_encoding, min_value, min_encoding, zero_value, zero_encoding, negzero_value, negzero_encoding);

  return return_tuple;
  
}

// ___________________________________________________________
// *** [4] bool vector support functions ***
// convert_intvec_to_boolvec(.) - returns a Vec<bool> derived from a Vec<u8>
// convert_boolvec_to_intvec(.) - returns a Vec<u8> derived from a Vec<bool>


fn convert_intvec_to_boolvec(int_vec:&Vec<u8>) -> Vec<bool> {
  // returns a Vec<bool> derived from a Vec<u8>

  let mut bool_vec:Vec<bool> = vec![];
  
  for entry in int_vec {
    if *entry == 1 {
      bool_vec.push(true);
    } else {
      bool_vec.push(false);
    }
  }
  return bool_vec;
}

fn convert_boolvec_to_intvec(bool_vec:&Vec<bool>) -> Vec<u8> {
  // returns a Vec<u8> derived from a Vec<bool>

  let mut int_vec:Vec<u8> = vec![];
  
  for entry in bool_vec {
    if *entry {
      int_vec.push(1);
    } else {
      int_vec.push(0);
    }
  }
  return int_vec;
}

// ______________________________________
// *** [5] bitwise operators ***
// bitwise_and(.)
// bitwise_and_inplace(.)
// bitwise_or(.)
// bitwise_or_inplace(.)
// bitwise_xor(.)
// bitwise_xor_inplace(.)
// bitwise_not(.)
// bitwise_not_inplace(.)
// bitwise_leftshift(.)
// bitwise_leftshift_inplace(.)
// bitwise_rightshift(.)
// bitwise_rightshift_inplace(.)

fn bitwise_and(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // applies bitwise AND (&) to two bool vectors
  // assumes equivalent length inputs
  // an alternate form may apply wrapper to this function to add support for variable length inputs
  // for now will return a newly populated vector
  // an alternate form may apply inplace conversion to one of the vectors for efficiency

  let mut output_vector:Vec<bool> = vec![];
  for (i, entry) in vector_one.iter().enumerate() {
    output_vector.push(*entry && vector_two[i]);
  }
  return output_vector;
}

fn bitwise_and_inplace(vector_one:&mut Vec<bool>, vector_two:&Vec<bool>) {
  // applies inplace bitwise AND (&) to two bool vectors
  // assumes equivalent length inputs
  // an alternate form may apply wrapper to this function to add support for variable length inputs
  // ***the result of operation populated in vector_one***

  for i in 0..vector_one.len() {
    vector_one[i] = vector_one[i] && vector_two[i];
  }
}

fn bitwise_or(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // applies bitwise OR (|) to two bool vectors
  // assumes equivalent length inputs
  // an alternate form may apply wrapper to this function to add support for variable length inputs
  // for now will return a newly populated vector
  // an alternate form may apply inplace conversion to one of the vectors for efficiency

  let mut output_vector:Vec<bool> = vec![];
  for (i, entry) in vector_one.iter().enumerate() {
    output_vector.push(entry | vector_two[i]);
  }
  return output_vector;
}

fn bitwise_or_inplace(vector_one:&mut Vec<bool>, vector_two:&Vec<bool>) {
  // applies inplace bitwise OR (|) to two bool vectors
  // assumes equivalent length inputs
  // an alternate form may apply wrapper to this function to add support for variable length inputs
  // ***the result of operation populated in vector_one***

  for i in 0..vector_one.len() {
    vector_one[i] = vector_one[i] | vector_two[i];
  }
}

fn bitwise_xor(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // applies bitwise XOR (^) to two bool vectors
  // assumes equivalent length inputs
  // an alternate form may apply wrapper to this function to add support for variable length inputs
  // for now will return a newly populated vector
  // an alternate form may apply inplace conversion to one of the vectors for efficiency

  let mut output_vector:Vec<bool> = vec![];
  for (i, entry) in vector_one.iter().enumerate() {
    output_vector.push(entry ^ vector_two[i]);
  }
  return output_vector;
}

fn bitwise_xor_inplace(vector_one:&mut Vec<bool>, vector_two:&Vec<bool>) {
  // applies inplace bitwise XOR (^) to two bool vectors
  // assumes equivalent length inputs
  // an alternate form may apply wrapper to this function to add support for variable length inputs
  // ***the result of operation populated in vector_one***

  for i in 0..vector_one.len() {
    vector_one[i] = vector_one[i] ^ vector_two[i];
  }
}

fn bitwise_not(vector_one:&Vec<bool>) -> Vec<bool> {
  // applies bitwise NOT (^) to one bool vector
  // for now will return a newly populated vector
  // an alternate form may apply inplace conversion to one of the vectors for efficiency

  let mut output_vector:Vec<bool> = vec![];
  for entry in vector_one.iter() {
    output_vector.push(!entry);
  }
  return output_vector;
}

fn bitwise_not_inplace(vector_one:&mut Vec<bool>) {
  // applies inplace bitwise NOT (^) to one bool vector
  // translates vector inplace without a return object

  for i in 0..vector_one.len() {
    vector_one[i] = !vector_one[i];
  }
}

fn bitwise_leftshift(vector_one:&Vec<bool>) -> Vec<bool> {
  // applies bitwise LEFTSHIFT (<<) to one bool vector
  // for now will return a newly populated vector
  // an alternate form may apply inplace conversion to one of the vectors for efficiency

  // for now assuming a leading temp bit may have been inserted eternally
  // if desired to avoid loss of precision

  let mut output_vector:Vec<bool> = vec![];
  for entry in vector_one[1..].iter() {
    output_vector.push(*entry);
  }
  output_vector.push(false);
  return output_vector;
}

fn bitwise_leftshift_inplace(vector_one:&mut Vec<bool>) {
  // applies inplace bitwise LEFTSHIFT (>>) to one bool vector
  // translates vector inplace without a return object

  let vector_one_len = vector_one.len();
  for i in 0..vector_one_len {
    if i < vector_one_len - 1 {
      vector_one[i] = vector_one[i+1];
    } else {
      vector_one[i] = false;
    }
  }
}

fn bitwise_rightshift(vector_one:&Vec<bool>) -> Vec<bool> {
  // applies bitwise RIGHTSHIFT (>>) to one bool vector
  // for now will return a newly populated vector
  // an alternate form may apply inplace conversion to one of the vectors for efficiency

  // note that this pads left side
  // so that returned length is same
  // assumes that if desired
  // a trailing null bit may have been inserted externally
  // if desired to retain precision

  let mut output_vector:Vec<bool> = vec![false];
  let vector_one_length = vector_one.len() - 1;

  for entry in vector_one[0..vector_one_length].iter() {
    output_vector.push(*entry);
  }
  return output_vector;
}

fn bitwise_rightshift_inplace(vector_one:&mut Vec<bool>) {
  // applies inplace bitwise RIGHTSHIFT (>>) to one bool vector
  // translates vector inplace without a return object

  let vector_one_len = vector_one.len();
  for i in 0..vector_one_len {
    if i < vector_one_len - 1 {
      vector_one[vector_one_len - i -1] = vector_one[vector_one_len - i -2];
    } else {
      vector_one[vector_one_len - i -1] = false;
    }
  }
}

// ______________________________________
// *** [6] integer operations
// hat tip for a few tutorials from iq.opengenus.org
// integer_convert(.) - for use to validate the integer arithmetic operations
// integer_revert(.) - translates a u32 to Vec<bool>
// integer_add(.)
// integer_add_entended(.) - several options for variations on integer_add
// integer_increment(.) - increments vector encoding by 1
// integer_decrement(.) - decrements vector encoding by 1
// integer_subtract(.)
// integer_multiply(.)
// integer_greaterthan(.)
// integer_divide(.)
// validate_bitwise_and_integer_operations(.) - temporary support function
// signedinteger_convert(.)
// signedinteger_revert(.)
// signedinteger_add(.)
// signedinteger_subtract(.)
// signedinteger_multiply(.)
// signedinteger_divide(.)
// signedinteger_greaterthan(.)
// signedinteger_abs(.)
// signedinteger_increment(.)
// signedinteger_decrement(.)


fn integer_convert(vector:&Vec<bool>) -> u32 {
  // for use to validate the integer arithmetic operations
  // assumes unsigned integer encoding in form 2^x+...2^2+2^1+2^0
  // and returns translation to u32

  let mut output:u32 = 0;
  let vector_length:u32 = vector.len() as u32;
  let mut power:u32 = 0;

  for (i, entry) in vector.iter().enumerate() {
    if *entry {
      power = u32::pow(2, vector_length - i as u32 - 1);
      output += power;
    }
  }
  return output;
}

fn integer_revert(integer:u32, bitwidth:u8) -> Vec<bool> {
  // for use to validate the integer arithmetic operations
  // returns unsigned integer encoding in form 2^x+...2^2+2^1+2^0

  let mut output:Vec<bool> = vec![false; bitwidth as usize];
  let mut integer_copy:u32 = integer.clone();
    
  for i in 0..bitwidth {
    if integer_copy >= u32::pow(2, bitwidth as u32 - i as u32 - 1) {
      output[i as usize] = true;
      integer_copy -= u32::pow(2, bitwidth as u32 - i as u32 - 1);
    }
  }
  return output;
}

fn integer_add(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // applies bitwise ADD (+) to two integer encoded bool vectors
  // assumes equivalent length inputs
  // assumes unsigned integer encoding in form 2^x+...2^2+2^1+2^0
  // assumes no sign bit
  // order of bits assumed as rightward decreasing register size
  // in current form don't have signal for overflow

  if !(vector_two.contains(&true)) {
    return vector_one.to_vec();
  } else {
    return integer_add( &bitwise_xor(&vector_one, &vector_two), &bitwise_leftshift( &bitwise_and(&vector_one, &vector_two) ) );
  }
}

fn integer_add_extended(vector_one:&Vec<bool>, vector_two:&Vec<bool>, append_leading_bits:bool, assumed_leading_bit:bool, rightside_aligned:bool) -> Vec<bool> {
  // the base integer_add assumes consistent width and doesn't accomodate overflow
  // this function extends integer_add to include following options:

  // append_leading_bits: when true, append two leading bits to each vector to prevent overflow
  // assumed_leading_bit: when true, assumes second of the leading bits for append_leading_bits is activated (as would be the case for mantissa)
  // automatically extends length for matched vector widths
  // rightside_aligned: if vectors are different lengths, if rightside_aligned pad false on left of shorter vector, else pad false on right
  // (for use with mantissas we can assume leftside alligned, note that append_leading_bits takes place after padding)

  // once those options are incorporated
  // we can pass to the base integer_add function

  let vector_one_len = vector_one.len() as u8;
  let vector_two_len = vector_two.len() as u8;  

  if !append_leading_bits {

    if vector_one_len == vector_two_len {
      return integer_add(&vector_one, &vector_two);
      
    } else {

      let mut max_len:u8 = vector_one_len;
      if vector_two_len > vector_one_len {
        max_len = vector_two_len;
      }

      return integer_add(
        &(vector_padout(&vector_one, max_len, rightside_aligned)), 
        &(vector_padout(&vector_two, max_len, rightside_aligned))
      );
    }
  } else {
    // else append_leading_bits is true
    let mut leading_bits:Vec<bool> = vec![false, assumed_leading_bit];

    if vector_one_len == vector_two_len {
      
      return integer_add(
        &(vec![(*leading_bits).to_vec(), vector_one.to_vec()].concat()), 
        &(vec![leading_bits, vector_two.to_vec()].concat())
      );
      
    } else {
      // else vectors are diferent lengths
      // add concat of leading btis with padded out vector

      let mut max_len:u8 = vector_one_len;
      if vector_two_len > vector_one_len {
        max_len = vector_two_len;
      }
      
      return integer_add(
        &(vec![(*leading_bits).to_vec(), 
vector_padout(&vector_one, max_len, rightside_aligned)].concat()),
        &(vec![leading_bits, 
vector_padout(&vector_two, max_len, rightside_aligned)].concat())
      );
    }
  }

  fn vector_padout(input_vector:&Vec<bool>, max_len:u8, rightside_aligned:bool) -> Vec<bool> {
    // accepts a single vector input
    // a vector of u8's representing adjacent vector lengths
    // and a rightside_alighned bool indicating whether padding should be applied on left or right side
    // pads out the vector with null entries to match the max length found in len_vec
    // (note that we can expect for mantissa case that rightside_aligned is false)
    // (note that leading overflow bits handled superately in integer-add)

    let input_vector_len:u8 = input_vector.len() as u8;

    if (input_vector_len as u8) < max_len {
      // perform padout

      let pad_vec:Vec<bool> = vec![false; (max_len - input_vector_len) as usize];

      if rightside_aligned {

        return vec![pad_vec, input_vector.to_vec()].concat();
        
      } else {

        return vec![input_vector.to_vec(), pad_vec].concat();
        
      }

    } else {

      return input_vector.to_vec();
      
    }
  }
}

fn integer_increment(vector_input:&Vec<bool>) -> Vec<bool> {
  // treats input as an integer encoding
  // and increments by one

  // doesn't currently detect overflow in increment operation
  // pending further thought

  let mut one_value:Vec<bool> = vec![false; vector_input.len() - 1];

  one_value.push(true);

  let output_vector:Vec<bool> = integer_add(&vector_input, &one_value);

  return output_vector;
}

fn integer_decrement(vector_input:&Vec<bool>) -> Vec<bool> {
  // treats input as an integer encoding
  // and decrements by one

  let mut one_value:Vec<bool> = vec![false; vector_input.len() - 1];

  one_value.push(true);

  let output_vector:Vec<bool> = integer_subtract(&vector_input, &one_value);

  return output_vector;
}

fn integer_subtract(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // applies bitwise SUBTRACT (-) to two integer encoded bool vectors
  // assumes equivalent length inputs
  // assumes unsigned integer encoding in form 2^x+...2^2+2^1+2^0
  // assumes no sign bit
  // order of bits assumed as rightward decreasing register size
  // in current form don't have signal for overflow

  if !(vector_two.contains(&true)) {
    return vector_one.to_vec();
  } else {

    return integer_subtract( &bitwise_xor(&vector_one, &vector_two), &bitwise_leftshift( &bitwise_and(  
 &bitwise_not( &vector_one ) , &vector_two) ) );
    
  }
}

fn integer_multiply(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // applies bitwise MULTIPLY (*) to two integer encoded bool vectors
  // assumes equivalent length inputs
  // assumes unsigned integer encoding in form 2^x+...2^2+2^1+2^0
  // using method 
  // while(b != 0): if(b is odd) {result = result + a;} a = a * 2; b = b/2;
  // since these are integer inputs we can test whether b is odd by rightmost entry
  // for now am assuming output vector has same shape as input vectors for simplicity
  // otherwise is less clear how to settle on convention

  let mut output_vector:Vec<bool> = vec![false; vector_one.len() as usize];

  let mut a_vector:Vec<bool> = vector_one.clone();
  let mut b_vector:Vec<bool> = vector_two.clone();
  
  while b_vector.contains(&true) {
    
    // if b is odd, result = result + a
    if b_vector[b_vector.len()-1] == true {
      output_vector = integer_add(&output_vector, &a_vector);
    }

    // a = a * 2
    a_vector = bitwise_leftshift(&a_vector);

    // b = b/2
    b_vector = bitwise_rightshift(&b_vector);
    
  }

  return output_vector;
  
}

fn integer_greaterthan(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> bool {
  // compares two integers of same width to determine (>=)
  // if vector_one >= vector_two, return true
  // assumes equivalent length inputs
  // assumes unsigned integer encoding in form 2^x+...2^2+2^1+2^0

  for (i, entry) in vector_one.iter().enumerate() {
    if *entry && !vector_two[i as usize] {
      return true;
    } else if !entry && vector_two[i as usize] {
      return false;
    }
  }
  return true;
}


// fn integer_divide(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
//   // applies integer division 
//   // assumes equivalent length inputs
//   // assumes unsigned integer encoding in form 2^x+...2^2+2^1+2^0
//   // (returns integer floor)
//   // vector_one divided by vector_two
//   // there might be a more elegant way to do this with e.g. bit shifts and whatnot
//   // this works as a starting point
//   // (a future extension may apply something inspired by mantissa_multiply)

//   if integer_greaterthan(&vector_one, &vector_two) && vector_two.contains(&true) {

//     let mut increment_vector:Vec<bool> = vec![false; vector_one.len() as usize];
//     increment_vector[vector_one.len() as usize - 1] = true;

//     // println!("in integer divide");
//     // println!("increment_vector {:?}", increment_vector);
//     // println!("vector_one {:?}", vector_one);
//     // println!("vector_two {:?}", vector_two);
//     // println!("");
    
    
//     return integer_add( &increment_vector, &integer_divide( &integer_subtract(&vector_one, &vector_two), &vector_two));

//   } else {

//     return vec![false; vector_one.len() as usize];
    
//   }

// }



fn integer_divide(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {

  let vector_one_len = vector_one.len();

  if integer_greaterthan(&vector_two, &vector_one) && vector_two != vector_one {
    // if two > one return zero
    return vec![false; vector_one_len];
  }

  // initialize support variables
  let mut output_vector:Vec<bool> = vec![false; vector_one_len];
  let mut leftshit_count:u8 = 1;

  let mut subtracted_vector_one:Vec<bool> = vector_one.clone();
  let mut shifted_vector_two:Vec<bool> = vector_two.clone();

  let mut increment_vector:Vec<bool> = vec![false; vector_one_len - 1];
  increment_vector.push(true);

  // we'll start by preparing support variables
  // such as to work our way from most to least significant bits
  for i in 0..vector_one.len() {

    let pre_shift_check_vec:Vec<bool> = bitwise_leftshift(&shifted_vector_two);

    if integer_greaterthan(&pre_shift_check_vec, &shifted_vector_two) && integer_greaterthan(&vector_one, &pre_shift_check_vec) {

      shifted_vector_two = pre_shift_check_vec.clone();
      leftshit_count += 1;

      increment_vector = bitwise_leftshift(&increment_vector);

    } else {
      break
    }

  }

  // now that we have support variables initialized
  // we can start preparing the output_vec
  // 
  for i in 0..leftshit_count {

    if integer_greaterthan(&subtracted_vector_one, &shifted_vector_two) {

      output_vector = integer_add(&output_vector, &increment_vector);

      subtracted_vector_one = integer_subtract(&subtracted_vector_one, &shifted_vector_two);

    }

    shifted_vector_two = bitwise_rightshift(&shifted_vector_two);

    increment_vector = bitwise_rightshift(&increment_vector);

    // if !increment_vector.contains(&true) {
    //   break
    // }

  }
  return output_vector;
}


fn signedinteger_convert(vector:&Vec<bool>) -> i32 {
  // convert a signed integer encoding to i32
  // assumes leading sign bit, other registers in regular integer convention

  if vector[0] {
    // negative sign case
    return -1 * integer_convert(&vector[1..].to_vec()) as i32;
  } else {
    return integer_convert(&vector[1..].to_vec()) as i32;
  }
}

fn signedinteger_revert(integer:i32, bitwidth:u8) -> Vec<bool> {
  // convert a signed integer encoding to i32
  // assumes leading sign bit, other registers in regular integer convention

  if i32::is_negative(integer) {
    // negative sign case
    return vec![vec![true], integer_revert((-integer) as u32, bitwidth - 1)].concat();
  } else {
    return vec![vec![false], integer_revert(integer as u32, bitwidth - 1)].concat();
  }
}


fn signedinteger_add(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {  
  // assumes leading sign bit, other registers in regular integer convention

  if !vector_one[0] && !vector_two[0] {
    // both values positive case
    // return integer_add with leading postiive sign bit

    return vec![
            vec![false],
            integer_add(&vector_one[1..].to_vec(),
                        &vector_two[1..].to_vec()
                        )
           ].concat();

  } else if vector_one[0] && vector_two[0] {
    // both values negative case
    // return integer_add with leading negative sign bit

    return vec![
            vec![true],
            integer_add(&vector_one[1..].to_vec(),
                        &vector_two[1..].to_vec()
                        )
           ].concat();

  } else if !vector_one[0] && vector_two[0] {
    // positive vector_one and negative vector_two case
    // return conditional on which abs(value) is greater

    if integer_greaterthan(
      &vector_one[1..].to_vec(),
      &vector_two[1..].to_vec()
      ) {
      // if one >= two, return one - two with leading positive sign bit

      return vec![
        vec![false],
        integer_subtract(&vector_one[1..].to_vec(),
                         &vector_two[1..].to_vec()
                    )
        ].concat();

    } else {
      // if one < two, return two - one with leading negative sign bit

      return vec![
        vec![true],
        integer_subtract(&vector_two[1..].to_vec(),
                         &vector_one[1..].to_vec()
                    )
        ].concat();

    }
  } else {
    // else if vector_one[0] && !vector_two[0]
    // negative vector_one and positive vector_two case
    // return conditional on which abs(value) is greater

    if integer_greaterthan(
      &vector_two[1..].to_vec(),
      &vector_one[1..].to_vec()
      ) {
      // if two >= one, return two - one with leading positive sign bit

      return vec![
        vec![false],
        integer_subtract(&vector_two[1..].to_vec(),
                         &vector_one[1..].to_vec()
                    )
        ].concat();

    } else {
      // if two < one, return one - two with leading negative sign bit

      return vec![
        vec![true],
        integer_subtract(&vector_one[1..].to_vec(),
                        &vector_two[1..].to_vec()
                    )
        ].concat();

    }
  }
}



fn signedinteger_subtract(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // assumes leading sign bit, other registers in regular integer convention

  return signedinteger_add(
          vector_one,
          &vec![vec![!vector_two[0]], vector_two[1..].to_vec()].concat()
          )
}


fn signedinteger_multiply(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // assumes leading sign bit, other registers in regular integer convention

  return vec![
    vec![vector_one[0] != vector_two[0]], // returns negative sign when one of inputs was negative
    integer_multiply(&vector_one[1..].to_vec(),
                     &vector_two[1..].to_vec()
                )
    ].concat();
}

fn signedinteger_divide(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // assumes leading sign bit, other registers in regular integer convention

  return vec![
    vec![vector_one[0] != vector_two[0]], // returns negative sign when one of inputs was negative
    integer_divide(&vector_one[1..].to_vec(),
                   &vector_two[1..].to_vec()
                )
    ].concat();
}


fn signedinteger_greaterthan(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> bool {
  // if vector_one >= vector_two, return true
  // assumes leading sign bit, other registers in regular integer convention

  if vector_one[0] {
    // if vector_one is positive

    if vector_two[0] {
      // if vector_one is positive and vector_two is positive

      return integer_greaterthan(&vector_one[1..].to_vec(), &vector_two[1..].to_vec());

    } else {
      // vector_one is positive and vector_two is negative
      return true;
    }

  } else {
    // if vector_one is negative

    if vector_two[0] {
      // if vector_one is negative and vector_two is positive

      return false;

    } else {
      // vector_one is negative and vector_two is negative
      return integer_greaterthan(&vector_two[1..].to_vec(), &vector_one[1..].to_vec());
    }

  }
}


fn signedinteger_abs(vector_one:&Vec<bool>) -> Vec<bool> {
  // returns absolute value of a signed integer
  // by resetting the leading bit to false

  if vector_one[0] {
    return vec![vec![false], vector_one[1..].to_vec()].concat()
  } else {
    return vector_one.to_vec();
  }
}


fn signedinteger_increment(vector_input:&Vec<bool>) -> Vec<bool> {
  // increments a signed integer by one

  if !vector_input[0] {
    // if sign bit is false (positive value)
    return vec![vec![false], integer_increment(&vector_input[1..].to_vec())].concat();
  } else {
    return vec![vec![true], integer_decrement(&vector_input[1..].to_vec())].concat();
  }
}


fn signedinteger_decrement(vector_input:&Vec<bool>) -> Vec<bool> {
  // increments a signed integer by one

  if !vector_input[0] {
    // if sign bit is false (positive value)
    return vec![vec![false], integer_decrement(&vector_input[1..].to_vec())].concat();
  } else {
    return vec![vec![true], integer_increment(&vector_input[1..].to_vec())].concat();
  }
}



  

  // let mut increment_vector:Vec<bool> = vec![false; vector_one.len() - 1];
  // increment_vector.push(true);


  // let mut vector_one_clone:Vec<bool> = vector_one.clone();
  // let mut vector_two_clone:Vec<bool> = vector_two.clone();

  // let vector_one_len = vector_one.len();

  // for i in 0..vector_one_len {

  //   let j = vector_one_len - 1 - i;

  //   if vector_two_clone.contains(&true) && output_vec.contains(&false) {

  //     let mut check_vector = vec![false; vector_one_len];
  //     check_vector[j] = true;

  //     if vector_two[j] && integer_greaterthan(&vector_one_clone, &check_vector) {
  
  //       output_vec = integer_add(&output_vec, &increment_vector);

  //       vector_one_clone = integer_subtract(&vector_one_clone, &check_vector);
        
  //     }

  //     increment_vector = bitwise_leftshift(&increment_vector);

  //     vector_two_clone = bitwise_leftshift(&vector_two_clone);

  //   } else {
  //     break
  //   }

  // }

//   return output_vec;
// }


fn validate_bitwise_and_integer_operations() {
  // ok want to run some experiments on operations conventions
  // let's initialize some random encodings 
  // values shown are based on 8/3 convention

  // SPECIAL ENCODINGS OFF SCENARIO
  // // we'll have default special encodings of empty vector for when not specified
  // let special_encodings = SpecialEncodings {
  //   nan : Vec::<bool>::new(), // equivalent to initializing as vec![] but assigns a type
  //   nan_equals_nan : false,
  //   infinity : Vec::<bool>::new(),
  //   neginfinity : Vec::<bool>::new(),
  //   // negzero : true, // not currently inspected
  // };

  let expwidth:u8 = 4;

  // SPECIAL ENCODINGS ON SCENARIO, FP8 CONVENTIONS
  // conventions for inf at saturation and nan at neg zero with bitwidth=8
  let special_encodings = SpecialEncodings {
    nan : vec![true, false, false, false, false, false, false, false], // this is otherwise negative zero
    nan_equals_nan : false,
    infinity : vec![false, true, true, true, true, true, true, true], // this is otherwise saturation
    neginfinity : vec![true, true, true, true, true, true, true, true], // this is otherwise negative saturation
    zero : vec![false, false, false, false, false, false, false, false], 
    negzero : Vec::<bool>::new(),
    // negzero : false, // not currently inspected
     stochastic_rounding : false,
  };
  
  let encoding_1:Vec<bool> = vec![false, false, true, false, true, false, false, false];
  let value_1:f32 = 12.;

  let encoding_2:Vec<bool> = vec![true, false, true, true, false, false, false, true];
  let value_2:f32 = -1.0625;

  let encoding_3:Vec<bool> = vec![false, false, false, true, false, false, true, true];
  let value_3:f32 = 0.296875;

  let encoding_4:Vec<bool> =  vec![true, false, false, true, true, false, false, true];
  let value_4:f32 = -0.390625;

  let encoding_5:Vec<bool> = vec![false, false, false, false, true, false, false, true];
  // let value_5:f32 = -0.390625;

  let mut encoding_6:Vec<bool> = vec![true, false, false, true, true, false, false, true];
  let mut encoding_7:Vec<bool> = vec![false, false, false, false, true, false, false, true];

  let mut encoding_8:Vec<bool> = vec![false, false, true, true, true, true, false, true];
  

  // validate bitwise_and
  let result_and:Vec<bool> = bitwise_and(&encoding_1, &encoding_2);
  println!("{:?}", encoding_1);
  println!("{:?}", encoding_2);
  println!("AND");
  println!("{:?}", result_and);
  println!("");
  

  // validate bitwise_or
  let result_or:Vec<bool> = bitwise_or(&encoding_1, &encoding_2);
  println!("{:?}", encoding_1);
  println!("{:?}", encoding_2);
  println!("OR");
  println!("{:?}", result_or);
  println!("");


  // validate bitwise_xor
  let result_xor:Vec<bool> = bitwise_xor(&encoding_1, &encoding_2);
  println!("{:?}", encoding_1);
  println!("{:?}", encoding_2);
  println!("XOR");
  println!("{:?}", result_xor);
  println!("");


  // validate bitwise_not
  let result_not:Vec<bool> = bitwise_not(&encoding_1);
  println!("{:?}", encoding_1);
  println!("NOT");
  println!("{:?}", result_not);
  println!("");


  // validate bitwise_leftshift
  let result_leftshift:Vec<bool> = bitwise_leftshift(&encoding_1);
  println!("{:?}", encoding_1);
  println!("LEFTSHIFT");
  println!("{:?}", result_leftshift);
  println!("");


  // validate bitwise_rightshift
  let result_rightshift:Vec<bool> = bitwise_rightshift(&encoding_1);
  println!("{:?}", encoding_1);
  println!("RIGHTSHIFT");
  println!("{:?}", result_rightshift);
  println!("");

  //
  println!("true in vec![false, false] = {}", vec![false, false].contains(&true));


  // validate integer_add
  let result_add:Vec<bool> = integer_add(&encoding_3, &encoding_4);
  println!("");
  println!("{:?}", encoding_3);
  println!("{:?}", encoding_4);
  println!("ADD");
  println!("{:?}", result_add);
  println!("ADD");
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&encoding_4));
  println!("{}", integer_convert(&result_add));
  println!("");

  // validate integer_subtract
  let result_subtract:Vec<bool> = integer_subtract(&encoding_3, &encoding_5);
  println!("");
  println!("{:?}", encoding_3);
  println!("{:?}", encoding_5);
  println!("SUBTRACT");
  println!("{:?}", result_subtract);
  println!("SUBTRACT");
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&encoding_5));
  println!("{}", integer_convert(&result_subtract));
  println!("");

  // validate integer_subtract
  let result_subtract2:Vec<bool> = integer_subtract(&encoding_5, &encoding_3);
  println!("");
  println!("{:?}", encoding_5);
  println!("{:?}", encoding_3);
  println!("SUBTRACT2");
  println!("{:?}", result_subtract2);
  println!("SUBTRACT2");
  println!("{}", integer_convert(&encoding_5));
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&result_subtract2));
  println!("");

  // //
  // let asdfasdf:Vec<u8> = vec![1,2,3];
  // println!("{}", asdfasdf[asdfasdf.len()-1]);

  // validate integer_multiply
  let result_multiply:Vec<bool> = integer_multiply(&encoding_3, &encoding_5);
  println!("");
  println!("{:?}", encoding_3);
  println!("{:?}", encoding_5);
  println!("MULTIPLY");
  println!("{:?}", result_multiply);

  println!("MULTIPLY");
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&encoding_5));
  println!("{}", integer_convert(&result_multiply));
  println!("");

  // validate integer_greaterthan
  let result_greaterthan:bool = integer_greaterthan(&encoding_3, &encoding_5);
  println!("");
  println!("{:?}", encoding_3);
  println!("{:?}", encoding_5);
  println!("GREATERTHAN");
  println!("{}", result_greaterthan);

  println!("GREATERTHAN");
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&encoding_5));
  println!("{}", result_greaterthan);
  println!("");

  // validate integer_greaterthan
  let result_greaterthan2:bool = integer_greaterthan(&encoding_5, &encoding_3);
  println!("");
  println!("{:?}", encoding_5);
  println!("{:?}", encoding_3);
  println!("GREATERTHAN2");
  println!("{}", result_greaterthan2);

  println!("GREATERTHAN2");
  println!("{}", integer_convert(&encoding_5));
  println!("{}", integer_convert(&encoding_3));
  println!("{}", result_greaterthan2);
  println!("");

  // validate integer_greaterthan
  let result_greaterthan3:bool = integer_greaterthan(&encoding_5, &encoding_5);
  println!("");
  println!("{:?}", encoding_5);
  println!("{:?}", encoding_5);
  println!("GREATERTHAN3");
  println!("{}", result_greaterthan3);

  println!("GREATERTHAN3");
  println!("{}", integer_convert(&encoding_5));
  println!("{}", integer_convert(&encoding_5));
  println!("{}", result_greaterthan3);
  println!("");

  // validate integer_divide
  let result_divide:Vec<bool> = integer_divide(&encoding_3, &encoding_5);
  println!("");
  println!("{:?}", encoding_3);
  println!("{:?}", encoding_5);
  println!("DIVIDE");
  println!("{:?}", result_divide);

  println!("DIVIDE");
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&encoding_5));
  println!("{}", integer_convert(&result_divide));
  println!("");

  // validate integer_divide2
  let result_divide2:Vec<bool> = integer_divide(&encoding_5, &encoding_3);
  println!("");
  println!("{:?}", encoding_5);
  println!("{:?}", encoding_3);
  println!("DIVIDE2");
  println!("{:?}", result_divide2);

  println!("DIVIDE2");
  println!("{}", integer_convert(&encoding_5));
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&result_divide2));
  println!("");

  // validate integer_divide3
  let result_divide3:Vec<bool> = integer_divide(&encoding_4, &encoding_3);
  println!("");
  println!("{:?}", encoding_4);
  println!("{:?}", encoding_3);
  println!("DIVIDE3");
  println!("{:?}", result_divide3);

  println!("DIVIDE3");
  println!("{}", integer_convert(&encoding_4));
  println!("{}", integer_convert(&encoding_3));
  println!("{}", integer_convert(&result_divide3));
  println!("");

  // validate bitwise_and_inplace
  println!("{:?}", encoding_6);
  println!("{:?}", encoding_7);
  bitwise_and_inplace(&mut encoding_6, &mut &encoding_7);
  println!("AND");
  println!("{:?}", encoding_6);
  println!("{:?}", encoding_7);
  println!("");
  

  // validate bitwise_or_inplace
  println!("{:?}", encoding_6);
  println!("{:?}", encoding_7);
  bitwise_or_inplace(&mut encoding_6, &mut &encoding_7);
  println!("OR");
  println!("{:?}", encoding_6);
  println!("{:?}", encoding_7);
  println!("");


  // validate bitwise_xor_inplace
  println!("{:?}", encoding_6);
  println!("{:?}", encoding_7);
  bitwise_xor_inplace(&mut encoding_6, &mut &encoding_7);
  println!("XOR");
  println!("{:?}", encoding_6);
  println!("{:?}", encoding_7);
  println!("");


  // validate bitwise_not_inplace
  println!("{:?}", encoding_6);
  bitwise_not_inplace(&mut encoding_6);
  println!("NOT");
  println!("{:?}", encoding_6);
  println!("");


  // validate bitwise_leftshift_inplace
  println!("{:?}", encoding_6);
  bitwise_leftshift_inplace(&mut encoding_6);
  println!("LEFTSHIFT");
  println!("{:?}", encoding_6);
  println!("");


  // validate bitwise_rightshift_inplace
  println!("RIGHTSHIFT");
  println!("{:?}", encoding_6);
  bitwise_rightshift_inplace(&mut encoding_6);
  println!("{:?}", encoding_6);
  println!("");

  // validate fp_add
  let recovered_value_1 = get_value(&encoding_3, expwidth, &special_encodings);
  let recovered_value_2 = get_value(&encoding_5, expwidth, &special_encodings);
  let result_fp_add:Vec<bool> = fp_add(&encoding_3, &encoding_5, expwidth, &special_encodings);
  let recovered_value_3 = get_value(&result_fp_add, expwidth, &special_encodings);
  println!("");
  println!("{:?}", encoding_3);
  println!("{:?}", encoding_5);
  println!("expwidth = {}", expwidth);
  println!("{}", recovered_value_1);
  println!("{}", recovered_value_2);
  println!("fp_add");
  println!("{:?}", result_fp_add);
  println!("{}", recovered_value_3);

  // validate fp_add2
  let recovered_value_a = get_value(&encoding_1, expwidth, &special_encodings);
  let recovered_value_b = get_value(&encoding_2, expwidth, &special_encodings);
  let result_fp_add2:Vec<bool> = fp_add(&encoding_1, &encoding_2, expwidth, &special_encodings);
  let recovered_value_c = get_value(&result_fp_add2, expwidth, &special_encodings);
  println!("");
  println!("{:?}", encoding_1);
  println!("{:?}", encoding_2);
  println!("expwidth = {}", expwidth);
  println!("{}", recovered_value_a);
  println!("{}", recovered_value_b);
  println!("fp_add");
  println!("{:?}", result_fp_add2);
  println!("{}", recovered_value_c);

  // validate fp_subtract
  let recovered_value_1_fp_subtract = get_value(&encoding_3, expwidth, &special_encodings);
  let recovered_value_2_fp_subtract = get_value(&encoding_5, expwidth, &special_encodings);
  let result_fp_subtract:Vec<bool> = fp_subtract(&encoding_3, &encoding_5, expwidth, &special_encodings);
  let recovered_value_3_fp_subtract = get_value(&result_fp_subtract, expwidth, &special_encodings);
  println!("");
  println!("{:?}", encoding_3);
  println!("{:?}", encoding_5);
  println!("expwidth = {}", expwidth);
  println!("{}", recovered_value_1_fp_subtract);
  println!("{}", recovered_value_2_fp_subtract);
  println!("fp_subtract");
  println!("{:?}", result_fp_subtract);
  println!("{}", recovered_value_3_fp_subtract);

  // validate fp_subtract2
  let recovered_value_a_fp_subtract = get_value(&encoding_1, expwidth, &special_encodings);
  let recovered_value_b_fp_subtract = get_value(&encoding_2, expwidth, &special_encodings);
  let result_fp_add2_fp_subtract:Vec<bool> = fp_subtract(&encoding_1, &encoding_2, expwidth, &special_encodings);
  let recovered_value_c_fp_subtract = get_value(&result_fp_add2_fp_subtract, expwidth, &special_encodings);
  println!("");
  println!("{:?}", encoding_1);
  println!("{:?}", encoding_2);
  println!("expwidth = {}", expwidth);
  println!("{}", recovered_value_a_fp_subtract);
  println!("{}", recovered_value_b_fp_subtract);
  println!("fp_subtract");
  println!("{:?}", result_fp_add2_fp_subtract);
  println!("{}", recovered_value_c_fp_subtract);



  // // SPECIAL ENCODINGS ON SCENARIO, FP8 CONVENTIONS
  // // conventions for inf at saturation and nan at neg zero with bitwidth=8
  // let special_encodings = SpecialEncodings {
  let one_vec:Vec<bool> = vec![false, false, false, true, true, false, false, false];
  let nan_vec:Vec<bool> = vec![true, false, false, false, false, false, false, false];
  let inf_vec:Vec<bool> = vec![false, true, true, true, true, true, true, true];
  let neg_inf_vec:Vec<bool> = vec![true, true, true, true, true, true, true, true];

  println!("inf plus neginf {:?}", fp_add(&inf_vec, &neg_inf_vec, expwidth, &special_encodings));
  println!("inf plus inf {:?}", fp_add(&inf_vec, &inf_vec, expwidth, &special_encodings));
  println!("neginf plus neginf {:?}", fp_add(&neg_inf_vec, &neg_inf_vec, expwidth, &special_encodings));
println!("1 plus nan {:?}", fp_add(&one_vec, &nan_vec, expwidth, &special_encodings));

  let inf_value:f32 = f32::INFINITY;
  let neg_inf_value:f32 = f32::NEG_INFINITY;
  let nan_value:f32 = f32::NAN;
  println!("inf times neginf {}", inf_value * neg_inf_value);
  println!("inf times inf {}", inf_value * inf_value);
  println!("neginf * neginf {}", neg_inf_value * neg_inf_value);
println!("inf times nan {}", inf_value * nan_value);

  println!("############################");

  println!("inf dividedby neginf {}", inf_value/neg_inf_value);
  println!("inf dividedby inf {}", inf_value/inf_value);
  println!("neginf dividedby neginf {}", neg_inf_value/neg_inf_value);
  println!("inf dividedby 1. {}", inf_value/1.);
  println!("inf dividedby -1. {}", inf_value/-1.);
  println!("inf dividedby 0. {}", inf_value/0.);
  println!("1. dividedby 0. {}", 1./0.);
  println!("1. dividedby -0. {}", 1./-0.);

  // println!("1 dividedby neginf {}", 1./value2);
  // println!("1 dividedby inf {}", 1./value1);

  // validate fp_multiply
  let recovered_value_a_fp_multiply = get_value(&encoding_8, expwidth, &special_encodings);
  let recovered_value_b_fp_multiply = get_value(&encoding_2, expwidth, &special_encodings);
  let result_fp_add2_fp_multiply:Vec<bool> = fp_multiply(&encoding_8, &encoding_2, expwidth, &special_encodings);
  let recovered_value_c_fp_multiply = get_value(&result_fp_add2_fp_multiply.to_vec(), expwidth, &special_encodings);
  println!("");
  println!("{:?}", encoding_8);
  println!("{:?}", encoding_2);
  println!("expwidth = {}", expwidth);
  println!("{}", recovered_value_a_fp_multiply);
  println!("{}", recovered_value_b_fp_multiply);
  println!("fp_multiply");
  println!("{:?}", result_fp_add2_fp_multiply);
  println!("{}", recovered_value_c_fp_multiply);


  println!("");
  println!("before precision convert {:?}", encoding_5);
  println!("input expwidth = {}", expwidth);
  println!("output expwidth = {}, bitwidth = {}", 6, 16);
  let fp_match_precision_result = fp_match_precision(&encoding_5, expwidth, 16, 6, &special_encodings);
  println!("before precision convert {:?}", encoding_5);
  println!("after precision convert {:?}", fp_match_precision_result);

  println!("");
  println!("before precision convert {:?}", encoding_5);
  println!("input expwidth = {}", expwidth);
  println!("output expwidth = {}, bitwidth = {}", 2, 5);
  let fp_match_precision_result2 = fp_match_precision(&encoding_5, expwidth, 5, 3, &special_encodings);
  println!("before precision convert {:?}", encoding_5);
  println!("after precision convert {:?}", fp_match_precision_result2);


  // // validate fp_multiply2
  // let recovered_value_a_fp_multiply2 = get_value(&encoding_2, expwidth, &special_encodings);
  // let recovered_value_b_fp_multiply2 = get_value(&encoding_2, expwidth, &special_encodings);
  // let result_fp_add2_fp_multiply2:Vec<bool> = fp_multiply(&encoding_2, &encoding_2, expwidth, &special_encodings);
  // let recovered_value_c_fp_multiply2 = get_value(&result_fp_add2_fp_multiply2, expwidth, &special_encodings);
  // println!("");
  // println!("{:?}", encoding_2);
  // println!("{:?}", encoding_2);
  // println!("expwidth = {}", expwidth);
  // println!("{}", recovered_value_a_fp_multiply2);
  // println!("{}", recovered_value_b_fp_multiply2);
  // println!("fp_multiply");
  // println!("{:?}", result_fp_add2_fp_multiply2);
  // println!("{}", recovered_value_c_fp_multiply2);


  // // trying to think about mantissa division
  // let mut divide_vec_1:Vec<bool> = vec![true, true, true, false, true, false];

  // let mut divide_vec_2:Vec<bool> = vec![true, false, false, false, true, false];

  // // trying to think about mantissa division
  // // max division case
  // let mut divide_vec_1:Vec<bool> = vec![true, true, true, true, true, true];

  // let mut divide_vec_2:Vec<bool> = vec![true, false, false, false, false, false];

  // trying to think about mantissa division
  // min division case
  let mut divide_vec_1:Vec<bool> = vec![true, false, false, false, false, false];

  let mut divide_vec_2:Vec<bool> = vec![true, true, true, true, true, true];

  // ok now let's try that with padding
  // for infinite precision
  // we want a larger mantissa output
  // so it makes sense to pad output
  // but division can be nonrepeating
  // so need some rounding convention
  //

  let mut pad_len:usize = divide_vec_1.len();

  for _i in 0..pad_len {
    divide_vec_1.push(false);
    divide_vec_2.insert(0, false);
  }

  let divide_result_12 = integer_divide(&divide_vec_1, &divide_vec_2);
  println!("");
  println!("pad_len = {}", pad_len);
  println!("a / b = c");
  println!("{:?}", divide_vec_1);
  println!("{:?}", divide_vec_2);
  println!("{:?}", divide_result_12);




  // // validate fp_divide
  // let recovered_value_a_fp_divide = get_value(&encoding_2, expwidth, &special_encodings);
  // let recovered_value_b_fp_divide = get_value(&encoding_1, expwidth, &special_encodings);
  // let result_fp_add2_fp_divide:Vec<bool> = fp_divide(&encoding_2, &encoding_1, expwidth, &special_encodings);
  // let recovered_value_c_fp_divide = get_value(&result_fp_add2_fp_divide, expwidth, &special_encodings);
  // println!("");
  // println!("{:?}", encoding_2);
  // println!("{:?}", encoding_1);
  // println!("expwidth = {}", expwidth);
  // println!("{}", recovered_value_a_fp_divide);
  // println!("{}", recovered_value_b_fp_divide);
  // println!("fp_divide");
  // println!("{:?}", result_fp_add2_fp_divide);
  // println!("{}", recovered_value_c_fp_divide);

  // validate fp_divide
  let recovered_value_a_fp_divide = get_value(&encoding_1, expwidth, &special_encodings);
  let recovered_value_b_fp_divide = get_value(&encoding_2, expwidth, &special_encodings);
  let result_fp_add2_fp_divide:Vec<bool> = fp_divide(&encoding_1, &encoding_2, expwidth, &special_encodings);
  let recovered_value_c_fp_divide = get_value(&result_fp_add2_fp_divide, expwidth, &special_encodings);
  println!("");
  println!("{:?}", encoding_1);
  println!("{:?}", encoding_2);
  println!("expwidth = {}", expwidth);
  println!("{}", recovered_value_a_fp_divide);
  println!("{}", recovered_value_b_fp_divide);
  println!("fp_divide");
  println!("{:?}", result_fp_add2_fp_divide);
  println!("{}", recovered_value_c_fp_divide);


}

// ______________________________________
// *** [7] floating point operations ***
// fp_add_variable_exp(.) - returns without rounding, variable expwidth basis 
// expand_exp_register(.)
// fp_add(.) returns without rounding, same expwidth basis
// match_exp(.)
// mantissa_add_with_exp_align(.)
// mantissa_subtract_with_exp_align(.)
// added_mantissa_exp_align(.)
// subtracted_mantissa_exp_align(.)
// fp_subtract(.) - (a wrapper for fp_add)
// mantissa_multiply(.) - support function for fp_multiply
// fp_multiply(.)
// fp_multiply_variable_exp(.)
// fp_divide(.)
// fp_abs(.)
// fp_greaterthan(.)
// fp_equals(.)
// further basics pending

fn fp_add_variable_exp(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> (Vec<bool>, u8) {
  // performs an fp_add on vectors one and two
  // and if the output is a saturation value
  // expands the exponent register and tries again
  // in order for this to work in context of special encodings
  // we'll need to inspect for encodings prior to calling fp_add
  // this also differs from fp_add in that 
  // it returns a tuple with second entry the resulting exp_width

  // when adding two numbers number_one >> 0 and number_two << 0
  // there will often be a rounding applied even in conditions of expanded exponent
  // as the returned value often won’t have capacity for lowest decimal values
  // trying o figure out how to incorporate stochastic rounding into this scenario
  // right now our stochastic rounding is primarily associated with updates to precision
  // (in the fp_match_precision function)
  // Such that when decreasing the size of a mantissa, the reduced bits are removed and lowest register either rounds up or down
  // One way to do this would be prior to the addition
  // Add rounding support bits to the mantissa
  // And then round them off after the addition
  // Tabling this for a future extension


  // _____________

  // first check or special encodings
  // in manner consistent with fp_add
  // with expection that we have to return a tuple


  // ok first we'll check inputs for special encodings
  // we'll have following conventions:
  // if either input is NaN, output is NaN
  // if INF plus NEG_INF, output is NaN
  // if INF plus INF, output is INF
  // if NEG_INF plus NEG_INF, output is NEG_INF
  // anything + (zero or negzero) = input 
  // zero + negzero = zero
  // negzero + negzero = negzero

  let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

  let special_value_result_two:f32 = special_encodings.check_for_reserved_encoding(vector_two);

  // if either fp is a special value, walk through scenarios
  // (when values aren't special encodings returned as 1.0f32)
  if !(special_value_result_one == 1.0f32 && special_value_result_two == 1.0f32) {

    // if either input is NaN, output is NaN
    if f32::is_nan(special_value_result_one) || f32::is_nan(special_value_result_two) {
      return ((*special_encodings.nan).to_vec(), expwidth);
    } else if special_value_result_one == f32::INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_two == f32::INFINITY {
        return ((*special_encodings.infinity).to_vec(), expwidth);
      } else if special_value_result_two == f32::NEG_INFINITY {
        return ((*special_encodings.nan).to_vec(), expwidth);
      } else {
        return ((*special_encodings.infinity).to_vec(), expwidth);
      }
      
    } else if special_value_result_one == f32::NEG_INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_two == f32::NEG_INFINITY {
        return ((*special_encodings.neginfinity).to_vec(), expwidth);
      } else if special_value_result_two == f32::INFINITY {
        return ((*special_encodings.nan).to_vec(), expwidth);
      } else {
        return ((*special_encodings.neginfinity).to_vec(), expwidth);
      }

    } else if special_value_result_two == f32::INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_one == f32::INFINITY {
        return ((*special_encodings.infinity).to_vec(), expwidth);
      } else if special_value_result_one == f32::NEG_INFINITY {
        return ((*special_encodings.nan).to_vec(), expwidth);
      } else {
        return ((*special_encodings.infinity).to_vec(), expwidth);
      }

    } else if special_value_result_two == f32::NEG_INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_one == f32::NEG_INFINITY {
        return ((*special_encodings.neginfinity).to_vec(), expwidth);
      } else if special_value_result_one == f32::INFINITY {
        return ((*special_encodings.nan).to_vec(), expwidth);
      } else {
        return ((*special_encodings.neginfinity).to_vec(), expwidth);
      }


      
    } else if (special_value_result_one == -0.) && (special_value_result_two == -0.) {
      return ((*special_encodings.negzero).to_vec(), expwidth);
    } else if (special_value_result_one == 0. || special_value_result_one == -0.) && (special_value_result_two == 0. || special_value_result_two == -0.) {
      return ((*special_encodings.zero).to_vec(), expwidth);
    } else if special_value_result_one == 0. || special_value_result_one == -0. {
        return (vector_two.to_vec(), expwidth);
    }  else if special_value_result_two == 0. || special_value_result_two == -0. {
        return (vector_one.to_vec(), expwidth);
    }
  }
  // ___________

  // great now can call fp_add
  let returned_vec:Vec<bool> = fp_add(vector_one, vector_two, expwidth, special_encodings);

  let mut is_not_saturated:bool = true;

  if !returned_vec[1..].contains(&false) || !returned_vec[1..].contains(&true) {
    
    is_not_saturated = false;
    
  }

  if is_not_saturated {

    return (returned_vec, expwidth);

  } else {

    // initialize expanded exponent register
    let mut adjusted_exp_one:Vec<bool> = vector_one[1..(expwidth as usize +1)].to_vec();

    let mut adjusted_exp_two:Vec<bool> = vector_two[1..(expwidth as usize +1)].to_vec();

    // derive the expanded form
    adjusted_exp_one = expand_exp_register(&adjusted_exp_one);

    adjusted_exp_two = expand_exp_register(&adjusted_exp_two);

    // adjusted exponent width is one larger
    let adusted_exp_width:u8 = expwidth + 1;

    //now aggregate the expanded encodings
    let adjusted_vector_one:Vec<bool> = vec![vec![vector_one[0]], adjusted_exp_one, vector_one[(1+expwidth as usize)..].to_vec().clone()].concat();

    let adjusted_vector_two:Vec<bool> = vec![vec![vector_two[0]], adjusted_exp_two, vector_two[(1+expwidth as usize)..].to_vec().clone()].concat();

    // now pass those adjustments, which won't be saturated, back to this function

    return fp_add_variable_exp(&adjusted_vector_one, &adjusted_vector_two, adusted_exp_width, &special_encodings)
 
  }
  
}


fn expand_exp_register(input_exp:&Vec<bool>) -> Vec<bool> {
  // accepts as input an exponent encoded as bool vector 
  // per the configurable FP conventions
  // and returns a corresponding / equal valued vector
  // adjusted to have an additional exponent register
  // and corresponding mantissa alignment to retain value
  // the returned encoding has an exponent width basis of (expwidth + 1)


  // when we add the additional exponent register
  // per FP conventions, that means that there will be a new offset value
  // so we will need to adjust the exponents accordingly
  // for example, if increasing exponent width from 4 to 5
  // the offset would be increased from 7 to 15
  // 3 to 4 => 3 to 7
  // 2 to 3 => 1 to 3
  // in other words, the delta is 2^(expwidth - 1)

  // let orig_offset:u32 = u32::pow(2, (expwidth as u32 - 1)) - 1;
  // let new_offset:u32 = u32::pow(2, (expwidth as u32)) - 1;
  // let offset_delta:u32 = new_offset - orig_offset;

  // // this is a littel cheaper
  // let offset_delta = u32::pow(2, expwidth as u32- 1);

  // so when we increase the offset 
  // that basically means if considering exp as a raw integer encoding
  // we need a bigger integer in the exponent
  // for an equivalent output

  // which in practice basically amounts to adding a leftside register
  // and then incrementing the resulting next to top register 

  let mut exp_vector:Vec<bool> = input_exp.clone();

  let increment_vector:Vec<bool> = vec![vec![false, true], vec![false; exp_vector.len()-1]].concat();

  exp_vector.insert(0, false);

  // this integer_add doesn't need overflow detection
  return integer_add(&exp_vector, &increment_vector);
}
  


fn fp_add(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // performs floating point addition
  // as a starting point will assume same format between inputs
  // the returned vector has consistent sign and exp widths, 
  // but may be extended bitwidth for purposes of infinite precision mantissa
  // such that an external rounding operation may be applied
  // or otherwise populated into a higher precision accumulator
  // special encodings have conventions per the comments below
  // note that since this may return a different mantissa width
  // we're operating on premise that recovery of special encodings
  // would take place with an external rounding operation


  // ___________

  // ok first we'll check inputs for special encodings
  // we'll have following conventions:
  // if either input is NaN, output is NaN
  // if INF plus NEG_INF, output is NaN
  // if INF plus INF, output is INF
  // if NEG_INF plus NEG_INF, output is NEG_INF
  // anything + (zero or negzero) = input 
  // zero + negzero = zero
  // negzero + negzero = negzero

  let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

  let special_value_result_two:f32 = special_encodings.check_for_reserved_encoding(vector_two);

  // if either fp is a special value, walk through scenarios
  // (when values aren't special encodings returned as 1.0f32)
  if !(special_value_result_one == 1.0f32 && special_value_result_two == 1.0f32) {

    // if either input is NaN, output is NaN
    if f32::is_nan(special_value_result_one) || f32::is_nan(special_value_result_two) {
      return (*special_encodings.nan).to_vec();
    } else if special_value_result_one == f32::INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_two == f32::INFINITY {
        return (*special_encodings.infinity).to_vec();
      } else if special_value_result_two == f32::NEG_INFINITY {
        return (*special_encodings.nan).to_vec();
      } else {
        return (*special_encodings.infinity).to_vec();
      }
      
    } else if special_value_result_one == f32::NEG_INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_two == f32::NEG_INFINITY {
        return (*special_encodings.neginfinity).to_vec();
      } else if special_value_result_two == f32::INFINITY {
        return (*special_encodings.nan).to_vec();
      } else {
        return (*special_encodings.neginfinity).to_vec();
      }

    } else if special_value_result_two == f32::INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_one == f32::INFINITY {
        return (*special_encodings.infinity).to_vec();
      } else if special_value_result_one == f32::NEG_INFINITY {
        return (*special_encodings.nan).to_vec();
      } else {
        return (*special_encodings.infinity).to_vec();
      }

    } else if special_value_result_two == f32::NEG_INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_one == f32::NEG_INFINITY {
        return (*special_encodings.neginfinity).to_vec();
      } else if special_value_result_one == f32::INFINITY {
        return (*special_encodings.nan).to_vec();
      } else {
        return (*special_encodings.neginfinity).to_vec();
      }


      
    } else if (special_value_result_one == -0.) && (special_value_result_two == -0.) {
      return (*special_encodings.negzero).to_vec();
    } else if (special_value_result_one == 0. || special_value_result_one == -0.) && (special_value_result_two == 0. || special_value_result_two == -0.) {
      return (*special_encodings.zero).to_vec();
    } else if special_value_result_one == 0. || special_value_result_one == -0. {
        return vector_two.to_vec();
    }  else if special_value_result_two == 0. || special_value_result_two == -0. {
        return vector_one.to_vec();
    }
  }
  // ___________

  // future extension:
  // since we've already addressed special values
  // we can pass an empty special_encopdings
  // to all fo the support functions to speed up a tad

  let bitwidth:u8 = vector_one.len() as u8;

  // ok now let's consider sign scenarios
  let mut one_positive:bool = !vector_one[0];
  let mut two_positive:bool = !vector_two[0];
  
  let mut output_sign_vec:Vec<bool> = vec![false]; //initialize false for positive

  // i know is silly to have two variables here
  // reason is because the test is for >=
  let mut abs_one_greaterthan_abs_two:bool = true; // intiialize
  let mut abs_two_greaterthan_abs_one:bool = true; // intiialize

  let mut returned_exp:Vec<bool> = vec![];
  let mut returned_mantissa:Vec<bool> = vec![];

  let exp_vector_one:Vec<bool> = vector_one[1..(expwidth+1) as usize].to_vec();
  let exp_vector_two:Vec<bool> = vector_two[1..(expwidth+1) as usize].to_vec();

  let mantissa_vector_one:Vec<bool> = vector_one[(expwidth+1) as usize..].to_vec();
  let mantissa_vector_two:Vec<bool> = vector_two[(expwidth+1) as usize..].to_vec();  
  
  if one_positive && two_positive {
    // both values positive case

    // println!("qwer");

    // output_sign_vec[0] = false; // already initialized for positive

    (returned_exp, returned_mantissa) = mantissa_add_with_exp_align(&exp_vector_one, &exp_vector_two, &mantissa_vector_one, &mantissa_vector_two, bitwidth);

  } else if !one_positive && !two_positive {
    // both values negative case

    output_sign_vec[0] = true; // negative case

    (returned_exp, returned_mantissa) = mantissa_add_with_exp_align(&exp_vector_one, &exp_vector_two, &mantissa_vector_one, &mantissa_vector_two, bitwidth);
    
  } else if one_positive && !two_positive {
    // one positive two negative case

    // println!("1111");

    // get abs of vector_one and vector_two
    let abs_vector_one = fp_abs(&vector_one, &special_encodings);
    let abs_vector_two = fp_abs(&vector_two, &special_encodings);

    // get output sign based on whether vector_one >= vector_two
    abs_one_greaterthan_abs_two = fp_greaterthan(&abs_vector_one, &abs_vector_two, expwidth, &special_encodings);

    // println!("abs_one_greaterthan_abs_two = {}", abs_one_greaterthan_abs_two);

    if abs_one_greaterthan_abs_two {

      // output_sign_vec[0] = false; // already initialized for positive value

      (returned_exp, returned_mantissa) = mantissa_subtract_with_exp_align(&exp_vector_one, &exp_vector_two, &mantissa_vector_one, &mantissa_vector_two, bitwidth);
      
    } else {

      output_sign_vec[0] = true; // negative value

      (returned_exp, returned_mantissa) = mantissa_subtract_with_exp_align(&exp_vector_two, &exp_vector_one, &mantissa_vector_two, &mantissa_vector_one, bitwidth);
      
    }
    
  } else if !one_positive && two_positive {

    // println!("!one_positive && two_positive");
    
    // get abs of vector_one and vector_two
    let abs_vector_one = fp_abs(&vector_one, &special_encodings);
    let abs_vector_two = fp_abs(&vector_two, &special_encodings);

    // println!("abs_vector_one {:?}", abs_vector_one);
    // println!("abs_vector_two {:?}", abs_vector_two);

    // get output sign based on whether vector_two >= vector_one
    abs_two_greaterthan_abs_one = fp_greaterthan(&abs_vector_two, &abs_vector_one, expwidth, &special_encodings);

    if abs_two_greaterthan_abs_one {

      // output_sign_vec[0] = false; // already initialized for positive value
      // println!("abs_two_greaterthan_abs_one, {}", abs_two_greaterthan_abs_one);

      // println!("this is the greater then lesser vector in subtraction:");
      // println!("{:?}", exp_vector_two);
      // println!("{:?}", exp_vector_one);
      // println!("mantissa_vector_two {:?}", mantissa_vector_two);
      // println!("mantissa_vector_one {:?}", mantissa_vector_one);
      // println!("");
      

      (returned_exp, returned_mantissa) = mantissa_subtract_with_exp_align(&exp_vector_two, &exp_vector_one, &mantissa_vector_two, &mantissa_vector_one, bitwidth);
      
    } else {

      output_sign_vec[0] = true; // negative value

      (returned_exp, returned_mantissa) = mantissa_subtract_with_exp_align(&exp_vector_one, &exp_vector_two, &mantissa_vector_one, &mantissa_vector_two, bitwidth);
      
    }
  }

  // if there are any trailing nulls beyond the original bitwidth
  // go ahead and strike
  let returned_mantissa_len:usize = returned_mantissa.len();
  // let extra_bits_count:usize = vector_one.len() - expwidth as usize - 1 - returned_mantissa_len;
  let mut delta_mantissa_len:usize = 0;
  let orig_mantissa_len:usize = vector_one.len() - expwidth as usize - 1;
  let mut extra_bits_count:u8 = 0;
  if returned_mantissa_len > orig_mantissa_len {
    delta_mantissa_len = returned_mantissa_len - orig_mantissa_len;

    for i in 0..delta_mantissa_len {
      if !returned_mantissa[returned_mantissa_len - 1 -i] {
        
        extra_bits_count += 1;

      } else {
        break
      }
    }
    
  }

  // print!("extra_bits_count = {}", extra_bits_count);

  if extra_bits_count > 0 {
    let reduced_mantissa_len:usize = returned_mantissa_len - extra_bits_count as usize;

    returned_mantissa = returned_mantissa[..reduced_mantissa_len].to_vec();

  }
  
  // for i in 0..extra_bits_count {

  //   if !returned_mantissa[returned_mantissa_len - 1 - i] {
  //     returned_mantissa.remove(returned_mantissa_len - 1 - i);
  //   }
  // }
  

  // let returned_vector:Vec<bool> = vec![output_sign_vec, returned_exp, returned_mantissa].concat();
  
  // // //assume rounding conducted externally
  // // // return the concatinated encoding
  // return returned_vector;

  return vec![output_sign_vec, returned_exp, returned_mantissa].concat();
}

// fn match_exp(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8) -> (Vec<bool>, Vec<bool>) {

fn match_exp(input_mantissa_vector_one:&Vec<bool>, input_mantissa_vector_two:&Vec<bool>, input_exp_vector_one:&Vec<bool>, input_exp_vector_two:&Vec<bool>) -> (Vec<bool>, Vec<bool>) {
  // returns (mantissa_one, mantissa_two)
  // with increased rightside entries as needed for matched precision
  // and with added leading slots for 2^1 and 2^0
  // returns an adjusted mantissa_two with same exp basis as vector_one
  // based on convention that exp_one >= exp_two
  // as may be inspected externally

  // let mut exp_one:Vec<bool> = vec![];
  // let mut exp_two:Vec<bool> = vec![];

  // let mut mantissa_one:Vec<bool> = vec![];
  // let mut mantissa_two:Vec<bool> = vec![];

  let mut exp_one:Vec<bool>;
  let mut exp_two:Vec<bool>;

  let mut mantissa_one:Vec<bool>;
  let mut mantissa_two:Vec<bool>;

  // let mut saturation_flag:bool = false;

  // intialize into variables one and two
  // based on max exp of input one and two
  // (such that if exp_two > exp_one, etc)
  if integer_greaterthan(&input_exp_vector_one, &input_exp_vector_two) {

    // to perform addition we'll want the same scale exponents
    exp_one = input_exp_vector_one.clone();
    exp_two = input_exp_vector_two.clone();
    // println!("intialized exp_one {:?}", exp_one);
    // println!("intialized exp_two {:?}", exp_two);

    // saturation_flag = !input_exp_vector_one.contains(&false);

    mantissa_one = input_mantissa_vector_one.clone();
    mantissa_two = input_mantissa_vector_two.clone();
    
  } else {

    // to perform addition we'll want the same scale exponents
    exp_one = input_exp_vector_two.clone();
    exp_two = input_exp_vector_one.clone();
    // println!("intialized exp_one {:?}", exp_one);
    // println!("intialized exp_two {:?}", exp_two);

    // saturation_flag = !input_exp_vector_two.contains(&false);

    mantissa_one = input_mantissa_vector_two.clone();
    mantissa_two = input_mantissa_vector_one.clone();
    
  }

  // the configurable FP8 format assumes a leading 2^0
  // we will add that for purposes of addition
  mantissa_one.insert(0, true);
  mantissa_two.insert(0, true);
  // mantissa_two.insert(0, false);

  //and will have convention of an additional leading 2^1
  // to support cases of e.g. 1.5+1.5 = 2^1+2^0
  mantissa_one.insert(0, false);
  mantissa_two.insert(0, false);

  let exp_delta = integer_subtract(&exp_one, &exp_two);

  // println!("exp_delta = {:?}", exp_delta);
  // println!("before matched precision");
  // println!("mantissa_one = {:?}", mantissa_one);
  // println!("mantissa_two = {:?}", mantissa_two);
  // println!("");

  if exp_delta.contains(&true) {
    // exp delta tells us how many right shifts of mantissa_two are needed
    let mut shift_count = integer_convert(&exp_delta);

    // println!("shift_count **** = {}", shift_count);

    // shift_count = (shift_count as f32).log2() as u32; // experiment

    // we'll pad the two mantissas on the right with this count prior to shifting
    for i in 0..shift_count {
      
      mantissa_one.push(false);
      mantissa_two.push(false);
    }

    // now apply the rightshift to mantissa_two
    for i in 0..shift_count {
      // in case the ADD function used in some form of recursion
      // will apply the shifts without inplace
      // I suspect otherwise the inplace version might be slightly more efficient
      // bitwise_rightshift_inplace(&mut mantissa_two);
      mantissa_two = bitwise_rightshift(&mantissa_two);
    }
    
  } 
  
  // I thought about testing for saturation conditions here
  // think it would be more appropriate in mantissa_add_with_exp_align
  // } else if saturation_flag {
    
    // !exp_one.contains(&false) {
    
    // if mantissa_one.contains(&true) || mantissa_two.contains(&true) {
    //   // this is the overflow scenario
    //   // return saturation

    //   println!("asdfdsasa 1234");

    //   mantissa_one = vec![vec![false, true], vec![true; input_mantissa_vector_two.len()]].concat();

    //   mantissa_two = mantissa_one.clone();
      
    // }
  // }

  // println!("after matched precision");
  // println!("mantissa_one = {:?}", mantissa_one);
  // println!("mantissa_two = {:?}", mantissa_two);
  // println!("");

  // println!("mantissa_one");
  // println!("{:?}", mantissa_one);
  // println!("mantissa_two");
  // println!("{:?}", mantissa_two);
  
  return (mantissa_one, mantissa_two);  
}

fn mantissa_add_with_exp_align(exp_vector_one:&Vec<bool>, exp_vector_two:&Vec<bool>, mantissa_vector_one:&Vec<bool>, mantissa_vector_two:&Vec<bool>, bitwidth:u8) -> (Vec<bool>, Vec<bool>) {
  // takes input of potentially misaligned exponent vectors
  // and corresponding mantissa vectors
  // aligns the exponents with adjustment to one of the mantissas
  // inserts a leading 01 register to the mantissas
  // performs vector addition
  // then passes the resulting exponent and mantissa to added_mantissa_exp_align to readjust the mantissa within scale of exponent
  // returns the output from added_mantissa_exp_align

  // currently saturation detection for addition takes place in either added_mantissa_exp_align or mantissa_add_with_exp_align
  // (where the former is called within the latter)
  // based on cases where an exponenent is to be incremented 
  // and it already contains all activated registers
  // when detected, the saturation_flag marker is set to true
  // and the returned tuples are provided with full activations 
  // in both exp and mantissa register

  // a future extension might simplify this function
  // by integration of integer_add_extended
  // which is kind of redundant with some of this scope
  // this function extends by also performing
  // correspondign adjustments to the exponent

  // let mut aligned_mantissa_one:Vec<bool> = vec![];
  // let mut aligned_mantissa_two:Vec<bool> = vec![];
  // let mut returned_mantissa:Vec<bool> = vec![];
  // let mut returned_exp:Vec<bool> = vec![];

  let mut aligned_mantissa_one:Vec<bool>;
  let mut aligned_mantissa_two:Vec<bool>;
  let mut returned_mantissa:Vec<bool>;
  let mut returned_exp:Vec<bool>;

  let mut saturation_flag:bool = false;

  // find largest exponent and align mantissas
  // note that match_exp includes inserting the leading 01 regisers
  if integer_greaterthan(&exp_vector_one, &exp_vector_two) {

    // println!("qwer2");
    
    let mut basis_exp:Vec<bool> = exp_vector_one.clone();
    
    (aligned_mantissa_one, aligned_mantissa_two) = match_exp(&mantissa_vector_one, &mantissa_vector_two, exp_vector_one, exp_vector_two);

    // println!("aligned_mantissa_one==mantissa_vector_one = {}", &aligned_mantissa_one==mantissa_vector_one);
    // println!("aligned_mantissa_two==mantissa_vector_two = {}", &aligned_mantissa_two==mantissa_vector_two);

    // now perform addition to the mantissas
    returned_mantissa = integer_add(&aligned_mantissa_one, &aligned_mantissa_two);

    // println!("returned_mantissa after integer_add = {:?}", returned_mantissa);
    // println!("aligned_mantissa_one after integer_add = {:?}", aligned_mantissa_one);
    
    // if !integer_greaterthan(&returned_mantissa[2..].to_vec(), &aligned_mantissa_one[2..].to_vec()) {

    if !integer_greaterthan(&returned_mantissa, &aligned_mantissa_one) {
      // if returned mantissa is less than either of the inputs
      // it means an overflow bit needs to be added to the exponent
      if basis_exp.contains(&false) {
        basis_exp = integer_increment(&basis_exp);  
      } else {
        saturation_flag = true;
      }
      

      // since performing a rightshift will pad right side
      // to retain precision
      returned_mantissa.push(false);
      
      returned_mantissa = bitwise_rightshift(&returned_mantissa);
      
    }
  
    // last step is to readjust exponent and mantissas
    // based on outcome of integer operation
    
    // returns tuple (returned_exp, returned_mantissa)
    return added_mantissa_exp_align(&basis_exp, &returned_mantissa, bitwidth);

  } else {

    // println!("asdfasdf");

    let mut basis_exp:Vec<bool> = exp_vector_two.clone();

    (aligned_mantissa_two, aligned_mantissa_one) = match_exp(&mantissa_vector_two, &mantissa_vector_one, exp_vector_two, exp_vector_one);
    
    // now perform addition to the mantissas
    returned_mantissa = integer_add(&aligned_mantissa_one, &aligned_mantissa_two);

    if !integer_greaterthan(&returned_mantissa[2..].to_vec(), &aligned_mantissa_one[2..].to_vec()) {
      // if returned mantissa is less than either of the inputs
      // it means an overflow bit needs to be added to the exponent

      if basis_exp.contains(&false) {
        basis_exp = integer_increment(&basis_exp);  
      } else {
        saturation_flag = true;
      }

      // since performing a rightshift will pad right side
      // to retain precision
      returned_mantissa.push(false);
      returned_mantissa = bitwise_rightshift(&returned_mantissa);
    }
  
    // last step is to readjust exponent and mantissas
    // based on outcome of integer operation

    // println!("saturation_flag = {}", saturation_flag);

    if !saturation_flag {
      // returns tuple (returned_exp, returned_mantissa)
      return added_mantissa_exp_align(&basis_exp, &returned_mantissa, bitwidth);
    } else {
      // else for overflow conditions return saturaiton
      let basis_exp_len = basis_exp.len();

      // *a variation could add additional exponent registers here
      
      return (basis_exp, vec![true; bitwidth as usize - basis_exp_len]);
    }
  }
}

fn mantissa_subtract_with_exp_align(exp_vector_one:&Vec<bool>, exp_vector_two:&Vec<bool>, mantissa_vector_one:&Vec<bool>, mantissa_vector_two:&Vec<bool>, bitwidth:u8) -> (Vec<bool>, Vec<bool>) {
  // resembles mantissa_add_with_exp_align
  // however in this case order matters
  // so we assume form of vector_one - vector_two
  // and we assume that fp_one > fp_two
  // (and if fp_two > fp_one we assume it was addressed external to this function)
  // returns the output from subtracted_mantissa_exp_align

  // let mut aligned_mantissa_one:Vec<bool> = vec![];
  // let mut aligned_mantissa_two:Vec<bool> = vec![];
  // let mut returned_mantissa:Vec<bool> = vec![];
  // let mut returned_exp:Vec<bool> = vec![];

  let mut aligned_mantissa_one:Vec<bool>;
  let mut aligned_mantissa_two:Vec<bool>;
  let mut returned_mantissa:Vec<bool>;
  let mut returned_exp:Vec<bool>;

  // align mantissas
  // note that match_exp includes inserting the leading 01 regisers
  // note that this assumes that fp_one >= fp_two validated externally

  (aligned_mantissa_one, aligned_mantissa_two) = match_exp(&mantissa_vector_one, &mantissa_vector_two, exp_vector_one, exp_vector_two);


  // now perform subtraction to the mantissas
  returned_mantissa = integer_subtract(&aligned_mantissa_one, &aligned_mantissa_two);

  // println!("returned_mantissa after integer subtract {:?}", returned_mantissa);
  // println!("");

  // // since we subtracted away the assumed register
  // // need to decrement exponent
  // let adjusted_exp:Vec<bool> = integer_decrement(&exp_vector_one);

  // last step is to readjust exponent and mantissas
  // based on outcome of integer operation
  
  // returns tuple (returned_exp, returned_mantissa)
  return subtracted_mantissa_exp_align(exp_vector_one, &returned_mantissa, bitwidth);

}

fn added_mantissa_exp_align(exp_vector:&Vec<bool>, mantissa_vector:&Vec<bool>, bitwidth:u8) -> (Vec<bool>, Vec<bool>) {
  // ok so the mantissa leading bit scenarios are {01, 00, 10, 11}
  // the context of this application is that we have an exponent register and a mantissa register
  // where the mantissa was derived from addition of preceding pair of mantissas of aligned exponent scale per exp_vector
  // using integer_add
  // and where we want to output a tuple of (exp, mantissa)
  // which were first adjusted to have mantissa config 01***_
  // by way of exponent increments / mantissa left shifts
  // or by way of exponenht decrements / mantissa rigth shifts
  // followed by dropping the first two mantissa registers
  // the preceding addition operation we assumed had mantissa inputs of 01***_
  // such that we know that the maximum was 11***_ without overflwo
  // **** we'll have convention that in cases of maxed out exponent, we halt the leftshift adjustments
  // and in cases of underflow in exponent, we return underflow saturation value
  // when performing mantissa right shifts we'll add a register
  // and when performing mantissa left shifts we'll retain same width (since we don't know original mantissa width herein)
  // the scenarios are:
  // 00 - if true in mantissa, mantissa leftshift / exponent increment until 01***, then drop leading two registers in mantissa. 
  // 01 - no adjustment, drop leading two mantissa registers
  // 10 or 11 - one mantissa rightshift, exponent decrement, drop two leading mantissa registers

  // currently saturation detection for addition takes place in either added_mantissa_exp_align or mantissa_add_with_exp_align
  // (where the former is called within the latter)
  // based on cases where an exponenent is to be incremented 
  // and it already contains all activated registers
  // when detected, the saturation_flag marker is set to true
  // and the returned tuples are provided with full activations 
  // in both exp and mantissa register

  // cool here it goes

  let mut saturation_flag:bool = false;

  let mut returned_exp:Vec<bool> = exp_vector.clone();
  let mut returned_mantissa:Vec<bool> = mantissa_vector.clone();

  // println!("mantissa_vector passed to added_exp_align");
  // println!("{:?}", mantissa_vector);
  // println!("exp_vector passed to added_exp_align");
  // println!("{:?}", exp_vector);
  // println!();

  
  if !mantissa_vector[0] && mantissa_vector[1] {
    // 01 - no adjustment, drop leading two mantissa registers

    returned_mantissa.remove(0);
    returned_mantissa.remove(0);

    return (returned_exp, returned_mantissa);
    
  } else if !mantissa_vector[0] && !mantissa_vector[1] {
    // 00 - if true in mantissa, mantissa leftshift / exponent increment until 01***, then drop leading two registers in mantissa.

    // if returned_mantissa.contains(&true) {

    let non_mantissa_bit_count = (returned_exp.len() + 1 - 2) as u8;
    let minimum_bit_count = bitwidth - non_mantissa_bit_count;

    let mut shiftcount:u8 = 0; 

    if (returned_mantissa[1..]).to_vec().contains(&true) {

      for entry in returned_mantissa[1..].iter() {
        if !entry {
          shiftcount +=1;
        } else {
          break
        }
      }
    }
    
    for _i in 0..shiftcount {
      // this is better than using a while loop
        
      if !returned_exp.contains(&false) {
        saturation_flag = true;
        
        // if exponent already maxed out, return overflow scenario
        // this is probably the reserved inf encoding
        returned_mantissa = vec![true; returned_mantissa.len()];
  
        return (returned_exp, returned_mantissa);
        
      } else if returned_mantissa.len() as u8 <= minimum_bit_count {

        // if mantissa width <= original mantissa width, etc
        returned_mantissa.push(false);
        
      }

      

      returned_mantissa.push(false);
      
      returned_mantissa = bitwise_leftshift(&returned_mantissa);

      returned_mantissa.remove(returned_mantissa.len() - 1);

      returned_exp = integer_increment(&returned_exp);
      
    }

    // // ok if after the for loop we still don't have a leading 01 in mantissa
    // // we'll go ahead and decrement the exponent
    // if !returned_mantissa[1] {

      
      
    //   returned_exp = integer_decrement(&returned_exp);
    // }      
      
    // }

    // now drop leading zeros
    returned_mantissa.remove(0);
    returned_mantissa.remove(0);

    return (returned_exp, returned_mantissa);

  } else {
    //if returned_mantissa[0] {
    // 10 or 11 - one mantissa rightshift, exponent decrement, drop two leading mantissa registers

    // println!("returned_exp {:?}", returned_exp);

    if !returned_exp.contains(&false) {

      // if exponent at maximum, return overflow scenario
      // this is probably the reserved inf encoding

      saturation_flag = true;
      
      returned_mantissa = vec![true; returned_mantissa.len()];

      return (returned_exp, returned_mantissa);

    }

    // since performing a rightshift will pad right side
    // to retain precision
    returned_mantissa.push(false);

    returned_mantissa = bitwise_rightshift(&returned_mantissa);

    returned_exp = integer_increment(&returned_exp);

    // now drop leading zeros
    returned_mantissa.remove(0);
    returned_mantissa.remove(0);

    return (returned_exp, returned_mantissa);

  }
  
}

fn subtracted_mantissa_exp_align(exp_vector:&Vec<bool>, mantissa_vector:&Vec<bool>, bitwidth:u8) -> (Vec<bool>, Vec<bool>) {
  // resembles added_mantissa_exp_align
  // for cases where mantissa_vector has leading bits 01 or 00
  // however treats cases of leading 1* bits differently
  // since assumes the subtraction had to borrow from exp

  // the scenarios are:
  // 00 - if true in mantissa, mantissa leftshift / exponent increment until 01***, then drop leading two registers in mantissa. 
  // 01 - no adjustment, drop leading two mantissa registers

  // ok am not positive, this is the tricky one
  // 10 or 11 - 1 mantissa rightshift, 2 exponent decrement, drop two leading mantissa registers

  // cool here it goes

  // println!("checkpoitn s1");

  let mut returned_exp:Vec<bool> = exp_vector.clone();
  let mut returned_mantissa:Vec<bool> = mantissa_vector.clone();

  
  if !mantissa_vector[0] && mantissa_vector[1] {
    // 01 - no adjustment, drop leading two mantissa registers

    // println!("checkpoitn s2");

    returned_mantissa.remove(0);
    returned_mantissa.remove(0);

    return (returned_exp, returned_mantissa);
    
  } else if !mantissa_vector[0] && !mantissa_vector[1] {
    // 00 - if true in mantissa, mantissa leftshift / exponent increment until 01***, then drop leading two registers in mantissa.

    // println!("checkpoitn s3");

    // if returned_mantissa.contains(&true) {

    let non_mantissa_bit_count = (returned_exp.len() + 1) as u8;
    let minimum_bit_count = bitwidth + 2 - non_mantissa_bit_count;

    // println!("before calculate shiftcount");
    // println!("returned_mantissa {:?}", returned_mantissa);
    // println!("returned_mantissa[1..(returned_mantissa.len() - minimum_bit_count as usize + 1)]");
    // println!("returned_mantissa.len() = {}", returned_mantissa.len());
    // println!("minimum_bit_count = {}", minimum_bit_count);
    
    // println!("(returned_mantissa.len() - minimum_bit_count as usize + 1) = {}", (returned_mantissa.len() - minimum_bit_count as usize + 1));
    // println!("{:?}", returned_mantissa[1..(returned_mantissa.len() - minimum_bit_count as usize + 1)]);
    
    let mut shiftcount:u8 = 0;

    if (returned_mantissa[1..]).to_vec().contains(&true) {

      for entry in returned_mantissa[1..].iter() {
  
        // println!("entry = {}", entry);
        // println!("!entry = {}", !entry);
          
  
      // for entry in returned_mantissa[0..(returned_mantissa.len() - minimum_bit_count as usize)].iter() {
        
        if !entry {
          shiftcount +=1;
        } else {
          break
        }
      }
      
    } else {
      // else if there wasn't a true in the mantissa
      // that means we're returning zero
      // for now we're treating zero as a special encoding
      // **I am not positive this scales to custom zero encodings
      // but for now am going to return the underflow saturation value

      returned_mantissa = vec![false; returned_mantissa.len()];

      returned_exp = vec![false; returned_exp.len()];

      return (returned_exp, returned_mantissa);
      
    }

    // println!("shiftcount {:?}", shiftcount);
    
    for _i in 0..shiftcount { // this is better than using a while loop
    
    
    // while !returned_mantissa[1] {
        
      if !returned_exp.contains(&true) {

        // println!("fdasfad");
        
        // if exponent already minimum, return underflow scenario
        // this is probably one of the reserved inf encoding
        returned_mantissa = vec![false; returned_mantissa.len()];
  
        return (returned_exp, returned_mantissa);
        
      } 
      // else if returned_mantissa.len() as u8 > minimum_bit_count {

      // println!("rewqrewq");

      // if mantissa width <= original mantissa width, etc
      // there's probably a cleaner way to do this than doing the addition every time
      // returned_mantissa.push(false);

      returned_mantissa.push(false);
    
      returned_mantissa = bitwise_leftshift(&returned_mantissa);

      returned_mantissa.remove(returned_mantissa.len()-1);

      // returned_exp = integer_increment(&returned_exp);

      returned_exp = integer_decrement(&returned_exp);

      // }
      
    }
    // println!("checkpoitn s5");
      // // ok if after the for loop we still don't have a leading 01 in mantissa
      // // we'll go ahead and decrement the exponent
      // if !returned_mantissa[1] {
      //   returned_exp = integer_decrement(&returned_exp);
      // }
      
    // }

    // now drop leading zeros
    returned_mantissa.remove(0);
    returned_mantissa.remove(0);

    return (returned_exp, returned_mantissa);

  } else {
    // if mantissa_vector[0] {
    // ok am not positive, this is the tricky one
    // 10 or 11 - 1 mantissa rightshift, 2 exponent decrement, drop two leading mantissa registers

    for i in 0..2 {

      if !returned_exp.contains(&true) {
        // if exponent at minimum, return underflow scenario
        // this is probably the reserved negative inf encoding
  
        // this is probably the reserved neginf encoding
        returned_mantissa = vec![false; returned_mantissa.len()];
  
        return (returned_exp, returned_mantissa);
  
      }

      returned_exp = integer_decrement(&returned_exp);

    }

    // since performing a rightshift will pad right side
    // to retain precision
    returned_mantissa.push(false);

    returned_mantissa = bitwise_rightshift(&returned_mantissa);

    // now drop leading zeros
    returned_mantissa.remove(0);
    returned_mantissa.remove(0);

    return (returned_exp, returned_mantissa);

  }

}

fn fp_subtract(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // a wrapper for fp_add function that merely swaps the sign register of vector_two

  let mut vector_two_subtract = vector_two.clone();
  vector_two_subtract[0] = !vector_two[0];
  
  
  let output_vector = fp_add(vector_one, 
                             &vector_two_subtract,
                             expwidth,
                             &special_encodings,
                            );

  return output_vector;
}

fn mantissa_multiply(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> Vec<bool> {
  // assumes mantissas received as 2^-1+2^-2+...
  // assumes consistent precision between inputs
  // (a future extension could rightside pad one of inputs in cases of different precision)
  // adds the assumed leftside 2^0 register
  // applies leftside padding based on orig width + 1
  // applies integer multiplication
  // scrub any trailing null activations beyond original width + 2
  // the resulting returned encoding has 
  // a returned first register of 2^1 and second register of 2^0
  // and may include additional trailing registers based on result of multiplication

  let orig_width:usize = vector_one.len();

  let mut mantissa_one:Vec<bool> = vector_one.clone();
  let mut mantissa_two:Vec<bool> = vector_two.clone();

  // add the assumed 2^0 register (now on leftside)
  mantissa_one.insert(0, true);
  mantissa_two.insert(0, true);

  // add leftside padding (the "+1" is the 2^1 register)
  for _i in 0..(orig_width + 1) {
    mantissa_one.insert(0, false);
    mantissa_two.insert(0, false);
  }

  // println!("baefore integer multiply");
  // println!("mantissa_one = {:?}", mantissa_one);
  // println!("mantissa_two = {:?}", mantissa_two);

  // perform integer multiplication
  let mut mantissa_product:Vec<bool> = integer_multiply(&mantissa_one, &mantissa_two);

  // println!("after integer multiply");
  // println!("mantissa_product = {:?}", mantissa_product);

  // // reverse order again
  // let mut output_vector:Vec<bool> = vec![];
  let mantissa_product_len = mantissa_product.len();

  for _i in 0..orig_width {

    let current_length = mantissa_product.len();
    
    if !mantissa_product[current_length - 1] {
      mantissa_product.remove(current_length - 1);
    } else {
      break
    }
  }

  // println!("after scrub trailing zeros");
  // println!("mantissa_product = {:?}", mantissa_product);

  // return output_vector;
  return mantissa_product;
}

// fn fp_multiply(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
//   // floating point multiply returning infinite precision
//   // based on assumption of consistent exponent width basis
//   // based on extracting sign bit, adding exponents, multiplying mantissas, 
//   // and then possible adjustment if resulting mantissa >= 2

//   // ___________

//   // ok first we'll check inputs for special encodings
//   // we'll have following conventions:
//   // if either input is NaN, output is NaN
//   // if INF * NEG_INF, output is NEG_INF
//   // if INF * INF, output is INF
//   // if NEG_INF * NEG_INF, output is INF
//   // any value or negzero times zero is zero
//   // anything else times negzero is negzero

//   let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

//   let special_value_result_two:f32 = special_encodings.check_for_reserved_encoding(vector_two);

//   // if either fp is a special value, walk through scenarios
//   // (when values aren't special encodings returned as 1.0f32)
//   if !(special_value_result_one == 1.0f32 && special_value_result_two == 1.0f32) {

//     // if either input is NaN, output is NaN
//     if f32::is_nan(special_value_result_one) || f32::is_nan(special_value_result_two) {
//       return (*special_encodings.nan).to_vec();
//     } else if special_value_result_one == f32::INFINITY {
//       // if INF times NEG_INF, output is NEG_INF
//       // if INF times INF, output is INF
//       // if NEG_INF times NEG_INF, output is INF
//       if special_value_result_two == f32::INFINITY {
//         return (*special_encodings.infinity).to_vec();
//       } else if special_value_result_two == f32::NEG_INFINITY {
//         return (*special_encodings.neginfinity).to_vec();
//       }
      
//     } else if special_value_result_one == f32::NEG_INFINITY {
//       // if INF plus NEG_INF, output is NaN
//       // if INF plus INF, output is INF
//       // if NEG_INF plus NEG_INF, output is NEG_INF
//       if special_value_result_two == f32::NEG_INFINITY {
//         return (*special_encodings.infinity).to_vec();
//       } else if special_value_result_two == f32::INFINITY {
//         return (*special_encodings.neginfinity).to_vec();
//       }
//     } else if special_value_result_one == 0. || special_value_result_two == 0. {
//       return (*special_encodings.zero).to_vec();
//     } else if special_value_result_one == -0. || special_value_result_two == -0. {
//       return (*special_encodings.negzero).to_vec();
//     }
//   }

//   // ___________

//   // get the returned sign bit
//   let returned_sign_bit:bool = vector_one[0] ^ vector_two[0]; // (based on XOR)

//   // get the returned exponent
//   // note that multiply has higher risk of overflow
//   // for now will just return saturation value, special encodings support pending
//   let mut returned_exp:Vec<bool> = integer_add(&vector_one[1..(expwidth as usize +1)].to_vec(), &vector_two[1..(expwidth as usize +1)].to_vec());

  

//   // note that if the returned vector is less than inputs
//   // it signals an exponent overflow scenario
//   if !integer_greaterthan(&returned_exp, &vector_one[1..(expwidth as usize +1)].to_vec()) || !integer_greaterthan(&returned_exp, &vector_two[1..(expwidth as usize +1)].to_vec()) {
//     // exponent overflow scenario
//     // return saturation value

//     println!("exponent overflow scenario");

//     // *****special encoding support for inf/neg_inf/nan pending
    
//     let mut returned_overflow_vec:Vec<bool> = vec![];

//     returned_overflow_vec.push(returned_sign_bit);

//     for _i in 0..(vector_one.len() - 1) {
//       returned_overflow_vec.push(true);
//     }
    
//     return returned_overflow_vec;
    
//   }

//   returned_exp = bitwise_rightshift(&returned_exp);

// // fn integer_greaterthan(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> bool {


//   // the *& means we are accessing the values from a slice of a vector 
//   // wrapped back to a vector by .to_vec()
//   let mut mantissa_one:Vec<bool> = (*&vector_one[(expwidth as usize +1)..]).to_vec();
//   let mut mantissa_two:Vec<bool> = (*&vector_two[(expwidth as usize +1)..]).to_vec();

//   let mut mantissa_product:Vec<bool> = mantissa_multiply(&mantissa_one, &mantissa_two);

//   println!("mantissa_one = {:?}", mantissa_one);
//   println!("mantissa_two = {:?}", mantissa_two);
//   println!("result of mantissa product is:");
//   println!("{:?}", mantissa_product);

//   // the returned mantissa includes added 2^1 and 2^0 registers
//   // we can expect that these registers will 
//   // either be in configuration of 01 or 10
//   // (since mantissa were recieved in range 1-1.99..)
//   // if in 01 configuraiton no adjustment to exponent needed
//   // if in 10 configuration, we'll add 1 to exponent and rightshift mantissa
//   if mantissa_product[0] {
//     let mut exponent_adder:Vec<bool> = vec![false; returned_exp.len()];

//     let exponent_adder_len = exponent_adder.len();
    
//     exponent_adder[exponent_adder_len-1] = true;

//     returned_exp = integer_add(&returned_exp, &exponent_adder);

//     // add extra bit to mantissa to retain precision
//     mantissa_product.push(false);

//     // now perform the rightshift
//     mantissa_product = bitwise_rightshift(&mantissa_product);
    
//   }

//   // now strip the leading registers from mantissa
//   mantissa_product.remove(0);
//   mantissa_product.remove(0);

//   // println!("returned_sign_bit = {}", returned_sign_bit);
//   // println!("returned_exp = {:?}", returned_exp);
//   // println!("mantissa_product = {:?}", mantissa_product);

//   // great now is just a matter of combining components

//   let mut output_vector:Vec<bool> = vec![];
  
//   output_vector.push(returned_sign_bit);

//   for entry in returned_exp.iter() {
//     output_vector.push(*entry);
//   }

//   for entry in mantissa_product.iter() {
//     output_vector.push(*entry);
//   }
  
//   return output_vector;
// }

fn fp_multiply(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // floating point multiply returning infinite precision
  // based on assumption of consistent exponent width basis
  // based on extracting sign bit, adding exponents, multiplying mantissas, 
  // and then possible adjustment if resulting mantissa >= 2
  
  // ___________

  // ok first we'll check inputs for special encodings
  // we'll have following conventions:
  // if either input is NaN, output is NaN
  // if INF * NEG_INF, output is NEG_INF
  // if INF * INF, output is INF
  // if NEG_INF * NEG_INF, output is INF
  // any value or negzero times zero is zero
  // anything else times negzero is negzero

  let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

  let special_value_result_two:f32 = special_encodings.check_for_reserved_encoding(vector_two);

  // if either fp is a special value, walk through scenarios
  // (when values aren't special encodings returned as 1.0f32)
  if !(special_value_result_one == 1.0f32 && special_value_result_two == 1.0f32) {

    // if either input is NaN, output is NaN
    if f32::is_nan(special_value_result_one) || f32::is_nan(special_value_result_two) {
      return (*special_encodings.nan).to_vec();
    } else if special_value_result_one == f32::INFINITY {
      // if INF times NEG_INF, output is NEG_INF
      // if INF times INF, output is INF
      // if NEG_INF times NEG_INF, output is INF
      if special_value_result_two == f32::INFINITY {
        return (*special_encodings.infinity).to_vec();
      } else if special_value_result_two == f32::NEG_INFINITY {
        return (*special_encodings.neginfinity).to_vec();
      }
      
    } else if special_value_result_one == f32::NEG_INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_two == f32::NEG_INFINITY {
        return (*special_encodings.infinity).to_vec();
      } else if special_value_result_two == f32::INFINITY {
        return (*special_encodings.neginfinity).to_vec();
      }
    } else if special_value_result_one == 0. || special_value_result_two == 0. {
      return (*special_encodings.zero).to_vec();
    } else if special_value_result_one == -0. || special_value_result_two == -0. {
      return (*special_encodings.negzero).to_vec();
    }
  }

  // ___________

  // get the returned sign bit
  let returned_sign_bit:bool = vector_one[0] ^ vector_two[0]; // (based on XOR)

  // get the returned exponent
  // note that multiply has higher risk of overflow
  // for now will just return saturation value, special encodings support pending
  // we'll add a leftside padding bit in case this overflows register (there is a subsequent offset subtraction)
  let mut returned_exp:Vec<bool> = integer_add(
    &vec![vec![false], vector_one[1..(expwidth as usize +1)].to_vec()].concat(), 
    &vec![vec![false], vector_two[1..(expwidth as usize +1)].to_vec()].concat()
  );
  

  // println!("");
  // println!("returned_exp from integer_add");
  // println!("{:?}", returned_exp);

  let mut exp_underflow:bool = false;
  let mut exp_overflow:bool = false;

  if !returned_exp[0] {

    // println!("checkpoint !returned_exp[0], exp_underflow = true");
    // possible underflow scenario
    // still pending result of offset subtraction
    // set conditional marker pending further inspection
    exp_underflow = true;
  }

  
  // println!("returned_exp");
  // println!("{:?}", returned_exp);
  // println!("");
  // println!("offset integer");
  // println!("{:?}", &(vec![vec![false; 2], vec![true; (expwidth as usize) - 1]].concat()));
  // println!("");

  // now we subtract the offset
  // (if the offset configuration was variable this might need to be adjustred)
  returned_exp = integer_subtract(&returned_exp, &(vec![vec![false; 2], vec![true; (expwidth as usize) - 1]].concat()));

  // println!("");
  // println!("returned_exp from integer_subtract of offset");
  // println!("{:?}", returned_exp);

  // note that if the returned exp vector has activation in top bit
  // it signals an exponent underflow scenario
  // otherwise we can just remove the leading bit

  

  
  if !returned_exp[0] {

    // println!("fp_multiply checkpoint exp_underflow = false");
    
    // non underflow scenario
    returned_exp.remove(0);

    exp_underflow = false;

  } else if exp_underflow {

    // println!("fp_multiply checkpoint exp_underflow = true");

    // println!("underflow scenario a");
    // exponent underflow scenario
    // return underflow 
    returned_exp = vec![false; expwidth as usize];

  } else {

    // println!("fp_multiply checkpoint exp_overflow = true");
    
    // println!("overflow scenario b");
    // exponent underflow scenario
    // return underflow 
    returned_exp = vec![true; expwidth as usize];
    exp_overflow = true;
  }

  // // now we subtract the offset
  // returned_exp = integer_subtract(&returned_exp, &(vec![vec![false; 2], vec![true; (expwidth as usize) - 1]].concat()));
  

  // // note that if the returned vector is less than inputs
  // // it signals an exponent overflow scenario
  // if !integer_greaterthan(&returned_exp, &vector_one[1..(expwidth as usize +1)].to_vec()) || !integer_greaterthan(&returned_exp, &vector_two[1..(expwidth as usize +1)].to_vec()) {
  //   // exponent overflow scenario
  //   // return saturation value

  //   println!("exponent overflow scenario");

  //   // *****special encoding support for inf/neg_inf/nan pending
    
  //   let mut returned_overflow_vec:Vec<bool> = vec![];

  //   returned_overflow_vec.push(returned_sign_bit);

  //   for _i in 0..(vector_one.len() - 1) {
  //     returned_overflow_vec.push(true);
  //   }
    
  //   return returned_overflow_vec;
    
  // }

  // returned_exp = bitwise_rightshift(&returned_exp);

  // returned_exp = integer_subtract(&returned_exp, )

// fn integer_greaterthan(vector_one:&Vec<bool>, vector_two:&Vec<bool>) -> bool {


  // the *& means we are accessing the values from a slice of a vector 
  // wrapped back to a vector by .to_vec()
  let mut mantissa_one:Vec<bool> = (*&vector_one[(expwidth as usize +1)..]).to_vec();
  let mut mantissa_two:Vec<bool> = (*&vector_two[(expwidth as usize +1)..]).to_vec();

  let mut mantissa_product:Vec<bool> = mantissa_multiply(&mantissa_one, &mantissa_two);

  // println!("mantissa_one = {:?}", mantissa_one);
  // println!("mantissa_two = {:?}", mantissa_two);
  // println!("result of mantissa product (including leading registers) is:");
  // println!("{:?}", mantissa_product);

  // the returned mantissa includes added 2^1 and 2^0 registers
  // we can expect that these registers will 
  // either be in configuration of 01 or 10
  // (since mantissa were recieved in range 1-1.99..)
  // if in 01 configuraiton no adjustment to exponent needed
  // if in 10 configuration, we'll add 1 to exponent and rightshift mantissa
  if mantissa_product[0] {
    let mut exponent_adder:Vec<bool> = vec![false; returned_exp.len()];

    let exponent_adder_len = exponent_adder.len();
    
    exponent_adder[exponent_adder_len-1] = true;

    if !exp_underflow {
      returned_exp = integer_add(&returned_exp, &exponent_adder);

      if !returned_exp.contains(&true) {
        exp_underflow = true;
      }
      
    }

    // add extra bit to mantissa to retain precision
    mantissa_product.push(false);

    // now perform the rightshift
    mantissa_product = bitwise_rightshift(&mantissa_product);
    
  }

  // now strip the leading registers from mantissa
  mantissa_product.remove(0);
  mantissa_product.remove(0);

  // println!("returned_sign_bit = {}", returned_sign_bit);
  // println!("returned_exp = {:?}", returned_exp);
  // println!("mantissa_product = {:?}", mantissa_product);

  // great now is just a matter of combining components

  if exp_underflow {

    // println!("return scenario 1");
    // return underflow
    return vec![false; vector_one.len()];

  } else if exp_overflow {

    // println!("return scenario 2");
    
    // return overflow
    return vec![vec![returned_sign_bit], vec![true; vector_one.len() - 1]].concat();
    
  } else {

    // println!("return scenario 3");

    let mut output_vector:Vec<bool> = vec![];
    
    output_vector.push(returned_sign_bit);
  
    for entry in returned_exp.iter() {
      output_vector.push(*entry);
    }
  
    for entry in mantissa_product.iter() {
      output_vector.push(*entry);
    }

    // println!("output_vector");
    // println!("{:?}", output_vector);
    // println!("");
    
    return output_vector;
    
  }
}


fn fp_multiply_variable_exp(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> (Vec<bool>, u8) {
  // performs an fp_add on vectors one and two
  // and if the output is a saturation value
  // expands the exponent register and tries again
  // in order for this to work in context of special encodings
  // we'll need to inspect for encodings prior to calling fp_add
  // this also differs from fp_add in that 
  // it returns a tuple with second entry the resulting exp_width

  // ___________

  // ok first we'll check inputs for special encodings
  // we'll have following conventions:
  // if either input is NaN, output is NaN
  // if INF * NEG_INF, output is NEG_INF
  // if INF * INF, output is INF
  // if NEG_INF * NEG_INF, output is INF
  // any value or negzero times zero is zero
  // anything else times negzero is negzero

  let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

  let special_value_result_two:f32 = special_encodings.check_for_reserved_encoding(vector_two);

  // if either fp is a special value, walk through scenarios
  // (when values aren't special encodings returned as 1.0f32)
  if !(special_value_result_one == 1.0f32 && special_value_result_two == 1.0f32) {

    // if either input is NaN, output is NaN
    if f32::is_nan(special_value_result_one) || f32::is_nan(special_value_result_two) {
      // return (*special_encodings.nan).to_vec();
      return ((*special_encodings.nan).to_vec(), expwidth);
    } else if special_value_result_one == f32::INFINITY {
      // if INF times NEG_INF, output is NEG_INF
      // if INF times INF, output is INF
      // if NEG_INF times NEG_INF, output is INF
      if special_value_result_two == f32::INFINITY {
        return ((*special_encodings.infinity).to_vec(), expwidth);
      } else if special_value_result_two == f32::NEG_INFINITY {
        return ((*special_encodings.neginfinity).to_vec(), expwidth);
      }
      
    } else if special_value_result_one == f32::NEG_INFINITY {
      // if INF plus NEG_INF, output is NaN
      // if INF plus INF, output is INF
      // if NEG_INF plus NEG_INF, output is NEG_INF
      if special_value_result_two == f32::NEG_INFINITY {
        return ((*special_encodings.infinity).to_vec(), expwidth);
      } else if special_value_result_two == f32::INFINITY {
        return ((*special_encodings.neginfinity).to_vec(), expwidth);
      }
    } else if special_value_result_one == 0. || special_value_result_two == 0. {
      return ((*special_encodings.zero).to_vec(), expwidth);
    } else if special_value_result_one == -0. || special_value_result_two == -0. {
      return ((*special_encodings.negzero).to_vec(), expwidth);
    }
  }

  // ___________

  // great now can call fp_add
  let returned_vec:Vec<bool> = fp_multiply(&vector_one, &vector_two, expwidth, &special_encodings);

  let mut is_not_saturated:bool = true;

  // println!("");
  // println!("within fp_multiply_variable_exp");
  // println!("");
  // println!("vector_one = {:?}", vector_one);
  // println!("{}", get_value(&vector_one, expwidth, &special_encodings));
  // println!("vector_two = {:?}", vector_two);
  // println!("{}", get_value(&vector_two, expwidth, &special_encodings));
  // println!("returned_vec from fp_multiply:");
  // println!("{:?}", returned_vec);

  if !returned_vec[1..].contains(&false) || !returned_vec[1..].contains(&true) {
    
    is_not_saturated = false;
    
  }

  if is_not_saturated {

    return (returned_vec, expwidth);

  } else {

    // initialize expanded exponent register
    let mut adjusted_exp_one:Vec<bool> = vector_one[1..(expwidth as usize +1)].to_vec();

    let mut adjusted_exp_two:Vec<bool> = vector_two[1..(expwidth as usize +1)].to_vec();

    // derive the expanded form
    adjusted_exp_one = expand_exp_register(&adjusted_exp_one);

    adjusted_exp_two = expand_exp_register(&adjusted_exp_two);

    // adjusted exponent width is one larger
    let adusted_exp_width:u8 = expwidth + 1;

    //now aggregate the expanded encodings
    let adjusted_vector_one:Vec<bool> = vec![vec![vector_one[0]], adjusted_exp_one, vector_one[(1+expwidth as usize)..].to_vec().clone()].concat();

    let adjusted_vector_two:Vec<bool> = vec![vec![vector_two[0]], adjusted_exp_two, vector_two[(1+expwidth as usize)..].to_vec().clone()].concat();

    // now pass those adjustments, which won't be saturated, back to this function

    return fp_multiply_variable_exp(&adjusted_vector_one, &adjusted_vector_two, adusted_exp_width, &special_encodings)
 
  }
  
}

fn fp_divide(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // needs more validation  but appears to be working
  
  
  // performs floating point division
  // with convention output = vector_one / vector_two
  // based on extracting sign bit, subtracting exponents, dividing mantissas, 
  // assumes same precision and format
  // and then possible adjustment if resulting mantissa <= 1
  // need to think about how best to implement, pending

  // special encoding support for this operation pending

  // // ___________

  // ok first we'll check inputs for special encodings
  // we'll have following conventions:

  // if either value is NaN return NaN
  // (inf or neginf) / (inf or neginf) returns NaN
  // inf dividedby value returns inf
  // inf dividedby negvalue returns neginf
  // anything dividedby 0. inf
  // anything dividedby -0. neginf

  let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

  let special_value_result_two:f32 = special_encodings.check_for_reserved_encoding(vector_two);

  // if either fp is a special value, walk through scenarios
  // (when values aren't special encodings returned as 1.0f32)
  if !(special_value_result_one == 1.0f32 && special_value_result_two == 1.0f32) {

    // if either input is NaN, output is NaN
    if f32::is_nan(special_value_result_one) || f32::is_nan(special_value_result_two) {
      return (*special_encodings.nan).to_vec();

    } else if special_value_result_two == 1.0f32 {
      // cases where numerator is special encoding, denominator is value
      if special_value_result_one == f32::INFINITY {
        return (*special_encodings.infinity).to_vec();
      } else if special_value_result_one == f32::NEG_INFINITY {
        return (*special_encodings.neginfinity).to_vec();
      } else if special_value_result_one == 0. {
        return (*special_encodings.zero).to_vec();
      } else if special_value_result_one == -0. {
        return (*special_encodings.negzero).to_vec();
      }
    } else if special_value_result_one == 1.0f32 {
      // cases where denominator is special encoding, numerator is value
      if special_value_result_one == f32::INFINITY {
        return (*special_encodings.zero).to_vec();
      } else if special_value_result_one == f32::NEG_INFINITY {
        return (*special_encodings.negzero).to_vec();
      } else if special_value_result_one == 0. {
        return (*special_encodings.nan).to_vec();
      } else if special_value_result_one == -0. {
        return (*special_encodings.nan).to_vec();
      }
    }
  }

  // // ___________

  // get the returned sign bit
  let returned_sign_bit:bool = vector_one[0] ^ vector_two[0]; // (based on XOR)

  // get the two exponents
  // and apply a leftside null pad to avoid underflow
  let mut exp_one:Vec<bool> = vec![vec![false], vector_one[1..(expwidth as usize +1)].to_vec().clone()].concat();

  let exp_two:Vec<bool> = vec![vec![false], vector_two[1..(expwidth as usize +1)].to_vec()].concat();

  // find out which exp is greater
  let exp_one_greater:bool = integer_greaterthan(&exp_one, &exp_two);

  // println!("exp_one_greater = {}", exp_one_greater);

  // find the delta betwee the exponents
  let mut exp_delta:Vec<bool> = vec![];
  
  if exp_one_greater {
    exp_delta = integer_subtract(&exp_one, &exp_two);
  } else {
    exp_delta = integer_subtract(&exp_two, &exp_one);
  }

  // println!("exp_delta = {:?}", exp_delta);

  // initialize returned exponent
  // may be further adjusted below based on mantissas
  // for now will either add or subtract the exp_delta from exp_one
  // and store result in the exp_one variable

  let offset_vector:Vec<bool> = vec![vec![false; 2], vec![true; (expwidth as usize) - 1]].concat();

  // println!("offset_vector = {:?}", offset_vector);

  let mut exp_overflow:bool = false;
  let mut exp_underflow:bool = false;
  
  // when exp_one_greater, the output will be bigger
  // so add to the exponent
  if exp_one_greater {
  // if !exp_one_greater {
    // println!("delta with integer_add case");
    // exp_one = integer_add(&exp_one, &exp_delta);
    exp_one = integer_add(&offset_vector, &exp_delta);

    if exp_one[0] {
      exp_overflow = true;
    }
    
  } else {
    // else the output will be smaller
    // so subtract from the exponent
    // println!("delta with integer_subtract case");
    // exp_one = integer_subtract(&exp_one, &exp_delta);
    exp_one = integer_subtract(&offset_vector, &exp_delta);

    if exp_one[0] {
      exp_underflow = true;
    }
    
  }

  // println!("exp_overflow = {}", exp_overflow);
  // println!("exp_underflow = {}", exp_underflow);

  // println!("exp_one after incorproating the delta");
  // println!("{:?}", exp_one);
  

  // // now we add the offset back in
  // // (if the offset configuration was variable this might need to be adjustred)
  // exp_one = integer_add(&exp_one, &(vec![vec![false; 2], vec![true; (expwidth as usize) - 1]].concat()));

  // println!("exp_one after adding the offset back on");
  // println!("{:?}", exp_one);

  // _____

  // great now we can look at mantissas

  // get the two mantissas
  let mut mantissa_one:Vec<bool> = vector_one[(expwidth as usize +1)..].to_vec().clone();

  let mut mantissa_two:Vec<bool> = vector_two[(expwidth as usize +1)..].to_vec().clone();

  // add the leading bit to the mantissas
  mantissa_one.insert(0, true);
  mantissa_two.insert(0, true);

  // println!("mantissa_one with leading pad bit");
  // println!("{:?}", mantissa_one);
  // println!("mantissa_two with leading pad bit");
  // println!("{:?}", mantissa_two);
  

  // note that our integer_divide function will return 0
  // for cases where mantissa_two > mantissa_one
  // so we'll pad out right side of mantissa_one
  // and left side of mantissa_two

  // can't do infinite precision for division
  // so just to pick a convention
  // will double the size of the registers
  // and then keep adding exponent increments 
  // until recover the original activation range
  // (every leftshift on mantissa bits equals exp bits decrement)

  let pad_len:usize = mantissa_one.len() - 1;
  // println!("pad_len = {}", pad_len);

  let pad_vector:Vec<bool> = vec![false; pad_len];

  // so we're going to divide mantissa_one / mantissa_two
  // to facilitate we will pad mantissa_one on right
  // and pad mantissa_two on left
  // mantissa_one = vec![mantissa_one, vec![false; pad_len], vec![false; pad_len]].concat();

  // mantissa_two = vec![vec![false; pad_len], mantissa_two, vec![false; pad_len]].concat();

  // want to try following
  // mantissa_one gets mantissa, 3 pads
  // mantissa_two gets 2 pads, mantissa, 1 pad

  // so for more precision here, I think it is justa a matter of increasing the pad size?


  // _______
  // this seems to work without full mantissa precision
  // note that this corresponds to this for loop below
  //  for _i in 0..(2 * pad_len) {
  
  // mantissa_one = vec![mantissa_one, vec![false; pad_len * 3]].concat();

  // mantissa_two = vec![vec![false; pad_len * 2], mantissa_two, vec![false; pad_len]].concat();

  // for larger pad sizes the memory appears to choke
  let mut pad_multiplier:usize = 4;
  if pad_len > 3 {
    if pad_len > 5 {
      pad_multiplier = 1;
    } else {
      pad_multiplier = 2;
    }
  }

  // ok this appears to work well
  // I think to increase the precision could increase this multiplier with corresponding increase to the for loop length below
  mantissa_one = vec![mantissa_one, vec![false; pad_len * pad_multiplier]].concat();

  mantissa_two = vec![vec![false; pad_len * pad_multiplier], mantissa_two].concat();
  
  // _______

  // // _______
  // // trying to figure out how to increase precision in mantissa registers
  // // so integer divide is going to keep number of registers and treat bottom regiser as a rightside aligned integer
  // // the purpose of the rightside / leftside padding
  // // is to support acccomodating both scenarios of one>two or two>one
  // // the thing is that fp_divide isn't really a critical item for infinite preciison in context of ML applications
  // // so might just table this and focus on higher order aggregations for now
  // mantissa_one = vec![mantissa_one, vec![false; pad_len * 3]].concat();

  // mantissa_two = vec![vec![false; pad_len * 2], mantissa_two, vec![false; pad_len]].concat();

  // // _______________

  // println!("mantissas after padding");
  // println!("mantissa_one {:?}", mantissa_one);
  // println!("mantissa_two {:?}", mantissa_two);

  // now divide the mantissas
  // store result in mantissa_one
  mantissa_one = integer_divide(&mantissa_one, &mantissa_two);

  // println!("after the mantissa division");
  // println!("exp_one");
  // println!("exp_one      {:?}", exp_one);
  // println!("mantissa_one {:?}", mantissa_one);
  

  // now we'll leftshift mantissa and increment exp
  // for the padding length
  // for _i in 0..pad_len { //(2 * pad_len) {
  // for _i in 0..3 { //(2 * pad_len) {
  for _i in 0..pad_len { //(2 * pad_len) {

    // println!("_i = {}", _i);

    // if exp_one.contains(&true) {

    if mantissa_one[0] {
      exp_one = integer_increment(&exp_one);
    }
      
    mantissa_one = bitwise_leftshift(&mantissa_one);

    mantissa_one.remove(mantissa_one.len() - 1);
  
      // exp_one = integer_decrement(&exp_one);
      
    // }
  // println!("");
  // println!("after the mantissa division");
  // println!("exp_one");
  // println!("exp_one      {:?}", exp_one);
  // println!("mantissa_one {:?}", mantissa_one);
    
  }

  // now that we've struck the pad_len padding
  // if the leading mantissa bit isn't active
  // we'll shift a few more registers with exp decrement
  if !mantissa_one[0] {
    for i in 0..pad_len {
      if !mantissa_one[0] {

        mantissa_one = bitwise_leftshift(&mantissa_one);

        mantissa_one.remove(mantissa_one.len() - 1);

        exp_one = integer_decrement(&exp_one);

      } else {
        break
      }
    }
  }

  // println!("");
  // println!("after the mantissa letshift, befroe strike leading bit");
  // // println!("exp_one");
  // println!("exp_one      {:?}", exp_one);
  // println!("mantissa_one {:?}", mantissa_one);

  // inspect the leading mantissa bit
  // if null then decrement exp one more time
  // and then strike leading mantissa bit
  if !mantissa_one[0] {

    // exp increment takes place at 2^0
    // the bias of exponent is based on 2^(expwidth-1) - 1
    // let exp_bias:u32 = u32::pow(2, expwidth as u32 - 1) - 1;

    // let exp_bias:u32 = 1;

    // println!("exp_bias = {}", exp_bias);

    // let exp_incrementer = integer_revert(&exp_bias, expwidth);

    // println!("exp_incrementer = {:?}", exp_incrementer);

    // exp_one = integer_add(&exp_one, &exp_incrementer);

    exp_one = integer_decrement(&exp_one);

  }

  mantissa_one.remove(0);

  // now check for exp overflow or underflow

  if exp_one[0] {
    if exp_one_greater {
      exp_overflow = true;
    } else {
      exp_underflow = true;
    }    
  }

  // println!("");
  // println!("after the leading bit strike");
  // println!("exp_one");
  // println!("exp_one      {:?}", exp_one);
  // println!("mantissa_one {:?}", mantissa_one);

  // now strike the leading pad bit in exponent
  exp_one.remove(0);

  if exp_underflow {
    // return underflow
    // right now this doesn't consider signed zero
    return vec![false; vector_one.len()];

  } else if exp_overflow {
    // return overflow
    return vec![vec![returned_sign_bit], vec![true; vector_one.len() - 1]].concat();
    
  } else {
    
    // combine sets to the output_vector
    let output_vector:Vec<bool> = vec![ vec![returned_sign_bit], exp_one, mantissa_one].concat();

    return output_vector;
  }
}

fn fp_abs(vector_one:&Vec<bool>, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // returns absolute value of floating point
  // assumes sign bit in first register

  // _______________
  
  // special encoding support for fp_abs:

  // abs(nan) = nan
  // abs(infinity) = inf
  // abs(neginfinity) = infinity **
  // abs(zero) = zero
  // abs(negzero) = zero ***

  let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

  // if either fp is a special value, walk through scenarios
  // (when values aren't special encodings returned as 1.0f32)
  if !(special_value_result_one == 1.0f32) {

    // if input is NaN, output is NaN
    if f32::is_nan(special_value_result_one) {
      // abs(nan) = nan
      return (*special_encodings.nan).to_vec();
    } else if special_value_result_one == f32::INFINITY {
      // abs(infinity) = inf
      return (*special_encodings.infinity).to_vec();
    } else if special_value_result_one == f32::NEG_INFINITY {
      // abs(neginfinity) = infinity **
      return (*special_encodings.infinity).to_vec();
    } else if special_value_result_one == 0. {
      // abs(zero) = zero
      return (*special_encodings.zero).to_vec();
    } else if special_value_result_one == -0. {
      // abs(negzero) = zero **
      return (*special_encodings.zero).to_vec();
    }
  }

  // _______________

  let mut output_vector:Vec<bool> = vector_one.clone();

  output_vector[0] = false;

  return output_vector;
}

fn fp_greaterthan(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> bool {
  // if vector_one >= vector_two returns true
  // assumes same precision basis
  // so workflow is base result on sign, then exp, then mantissa


  // _______________
  // fp_greaterthan special encodings support

  // nan >= nan (based on nan_equals_nan specification)
  // nan >= inf = false
  // nan >= -inf = false
  // nan >= zero = false
  // nan >= negzero = false
  // inf >= nan = false
  // inf >= inf = true
  // inf >= -inf = true
  // inf >= zero = true
  // inf >= negzero = true
  // -inf >= nan = false
  // -inf >= inf = false
  // -inf >= -inf = true
  // -inf >= zero = false
  // -inf >= negzero = false
  // zero >= nan = false
  // zero >= inf = false
  // zero >= -inf = true
  // zero >= zero = true
  // zero >= negzero = true
  // negzero >= nan = false
  // negzero >= inf = false
  // negzero >= -inf = true
  // negzero >= zero = true
  // negzero >= negzero = true

  let special_value_result_one:f32 = special_encodings.check_for_reserved_encoding(vector_one);

  let special_value_result_two:f32 = special_encodings.check_for_reserved_encoding(vector_two);

  // if either fp is a special value, walk through scenarios
  // (when values aren't special encodings returned as 1.0f32)
  if !(special_value_result_one == 1.0f32 && special_value_result_two == 1.0f32) {

    // if either input is NaN, output is false unless nan = nan
    if f32::is_nan(special_value_result_one) {

      if f32::is_nan(special_value_result_two) {
        if special_encodings.nan_equals_nan {
          return true;
        } else {
          return false;
        }
      } else if special_value_result_two == f32::INFINITY {
        return false;
      } else if special_value_result_two == f32::NEG_INFINITY {
        return false;
      } else if special_value_result_two == 0. {
        return false;
      } else if special_value_result_two == -0. {
        return false;
      }

    } else if special_value_result_one == f32::INFINITY {
      // inf >= nan = false
      // inf >= inf = true
      // inf >= -inf = true
      // inf >= zero = true
      // inf >= negzero = true

      if f32::is_nan(special_value_result_two) {
        return false;
      } else if special_value_result_two == f32::INFINITY {
        return true;
      } else if special_value_result_two == f32::NEG_INFINITY {
        return true;
      } else if special_value_result_two == 0. {
        return true;
      } else if special_value_result_two == -0. {
        return true;
      }
      
    } else if special_value_result_one == f32::NEG_INFINITY {
      // -inf >= nan = false
      // -inf >= inf = false
      // -inf >= -inf = true
      // -inf >= zero = false
      // -inf >= negzero = false

      if f32::is_nan(special_value_result_two) {
        return false;
      } else if special_value_result_two == f32::INFINITY {
        return false;
      } else if special_value_result_two == f32::NEG_INFINITY {
        return true;
      } else if special_value_result_two == 0. {
        return false;
      } else if special_value_result_two == -0. {
        return false;
      }

    } else if special_value_result_one == 0. {
      // zero >= nan = false
      // zero >= inf = false
      // zero >= -inf = true
      // zero >= zero = true
      // zero >= negzero = true

      if f32::is_nan(special_value_result_two) {
        return false;
      } else if special_value_result_two == f32::INFINITY {
        return false;
      } else if special_value_result_two == f32::NEG_INFINITY {
        return true;
      } else if special_value_result_two == 0. {
        return true;
      } else if special_value_result_two == -0. {
        return false;
      }

    } else if special_value_result_one == -0. {
      // negzero >= nan = false
      // negzero >= inf = false
      // negzero >= -inf = true
      // negzero >= zero = true
      // negzero >= negzero = true

      if f32::is_nan(special_value_result_two) {
        return false;
      } else if special_value_result_two == f32::INFINITY {
        return false;
      } else if special_value_result_two == f32::NEG_INFINITY {
        return true;
      } else if special_value_result_two == 0. {
        return true;
      } else if special_value_result_two == -0. {
        return true;
      }
    }
  }
  // _______________

  let mut returned_result = false;
  
  if !vector_one[0] && vector_two[0] {
    // if positive vector_one and negative vector_two
    // reurn true
    // println!("fp_gt one");

    returned_result = true;
    
  } else if vector_one[0] && !vector_two[0] {
    // if negative vector_one and positive vector_two
    // println!("fp_gt two");
    returned_result = false;
    
  } else {
    // println!("fp_gt three");
    // compare exponents

    // the equal case returns true
    if vector_one == vector_two {
      return true;
    }

    if integer_greaterthan(&vector_one[1..(expwidth as usize + 1)].to_vec(), &vector_two[1..(expwidth as usize + 1)].to_vec()) {

      if vector_one[1..(expwidth as usize + 1)] != vector_two[1..(expwidth as usize + 1)] {

        returned_result = true;
        
      } else {
        // else if exponents were equal, check the mantissa
        // note that this operation assumes equivalent precision between inputs
        // and will need to consider sign
        // at this pont have already established that signs are same

        let mantissa_one_greaterthan_two = integer_greaterthan(&vector_one[(expwidth as usize + 1)..].to_vec(), &vector_two[(expwidth as usize + 1)..].to_vec());

        if mantissa_one_greaterthan_two {
          if !vector_one[0] {
            // one > two and both values are positive
            return true;
          } else {
            // one > two and both values are negative
            return false;
          }
        } else {
          if !vector_one[0] {
            // two > one and both values are positive
            return false;
          } else {
            // two > one and both values are negative
            return true;
          }
        }
      }
    }
    //redundant
    // } else if &vector_one[1..(expwidth as usize + 1)] == &vector_two[1..(expwidth as usize + 1)] {
    //   // else if exponents are the same
    //   // compare the mantissas
    //   // noting that the result needs to consider sign

    //   if integer_greaterthan(&vector_one[(expwidth as usize + 1)..].to_vec(), &vector_two[(expwidth as usize + 1)..].to_vec()) {

    //     if vector_one[0] && vector_two[0] {
    //       // for positive integers with same exponent
    //       // a greater mantissa one returns true
    //       // (mixed sign case already considered above)
    //       returned_result = true;
    //     } else {
    //       returned_result = false;
    //     }

    //   } else {
    //     // this is case where mantissa_one < mantissa_two
    //     if vector_one[0] && vector_two[0] {
    //       // for positive integers with same exponent
    //       // a greater mantissa two returns false
    //       // (mixed sign case already considered above)
    //       returned_result = false;
    //     } else {
    //       returned_result = true;
    //     }
        
    //   }

    // }

    // if vector_one[0] && vector_two[0] {
    //   // if both are negative flip result
    //   returned_result = !returned_result;
    // }
    
  }



  return returned_result;

}

fn fp_equals(vector_one:&Vec<bool>, vector_two:&Vec<bool>, expwidth:u8, special_encodings:&SpecialEncodings) -> bool {
  // if vector_one == vector_two returns true
  // assumes same precision basis
  // primary purpose of this function is to accomodate special encodings

  if vector_one != vector_two {
    return false;
  } else {
    if !special_encodings.nan_equals_nan && f32::is_nan(special_encodings.check_for_reserved_encoding(vector_one)) && f32::is_nan(special_encodings.check_for_reserved_encoding(vector_two)) {
      // if both entries are nan, if not nan_equals_nan, return false
      return false;
    } else {
      // else results are equal
      return true
    }
  }
}

// ______________________________________
// *** [8] mixed precision support ***
// fp_match_precision(.) - converts vector_one precision to a given expwidth and bitwidth
// fp_convert_exp_precision(.)  - support function, expands or decreases exponent
// fp_convert_mantissa_precision(.) - support function, expands or decreases mantissa, rounding applied
// fp_increment_mantissa(.) - support function
// fp_compare(.)
// fp_fma(.)
// fp_set_scaler(.)
// fp_dot(.)

fn fp_match_precision(vector_input:&Vec<bool>, expwidth_one:u8, bitwidth_two:u8, expwidth_two:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // converts and returns vector_one to basis of vector_two
  // based on explicit exponent wdiths between the two
  // and explicit output bitwidth_two

  // -------
  // from a special encodings standpoing
  // it would be good to have convention that if input is a special encoding
  // we convert output to a similar special encoding
  // in alternate rpecision
  // the challenge is that special encoding specification
  // is currently based on a particular expwidth / bitwidth basis
  // so to add special encoding support we might need to rethink that convention

  // we'll have convention that if user passes bitwidth_two = 0
  // then bitwidth_two retains width achieved after exp conversion

  // println!("fp_match_precision checkpoint 1");
  // println!("");

  let mut output_vector = fp_convert_exp_precision(&vector_input, expwidth_one, expwidth_two, &special_encodings);

  // println!("fp_match_precision checkpoint 2");
  // println!("");

  // println!("output_vector = fp_convert_mantissa_precision(&output_vector, expwidth_two, bitwidth_two, &special_encodings);");
  // println!("output_vector = {:?}", output_vector);
  // println!("expwidth_two = {}", expwidth_two);
  // println!("bitwidth_two = {}", bitwidth_two);
  // println!("");

  // println!("before calling fp_convert_mantissa_precision");
  // println!("output_vector {:?}", output_vector);
  // println!("expwidth_two {}", expwidth_two);
  // println!("bitwidth_two {}", bitwidth_two);

  output_vector = fp_convert_mantissa_precision(&output_vector, expwidth_two, bitwidth_two, &special_encodings);

  // println!("fp_match_precision checkpoint 3");
  // println!("");

  return output_vector;
}


fn fp_convert_exp_precision(vector_input:&Vec<bool>, expwidth_input:u8, expwidth_output:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // ***validation pending***

  
  // for two exponent bases
  // vector_input recieved with expwidth_input basis
  // and returned with expwidth_output basis
  // assumes special encodings handled externally
  // (since in current form special encoding specification based on a specific format basis)
  // retains consistent mantissa width
  // in cases of overflow or underflow returns saturation

  // to retain consistent exponent value with offset
  // when increasing register count will increment next to top register after adding null in top register
  // and when decreasing register count will decrement next to top register after adding null in top register

  // sign and mantissa will be returned as is
  let mut output_sign:Vec<bool> = vec![];
  output_sign.push(vector_input[0]);

  let mut output_mantissa:Vec<bool> = (*&vector_input[(1 + expwidth_input as usize)..]).to_vec();

  let output_mantissa_len = output_mantissa.len();

  // we'll initialize output_exp with input_exp
  // and provide conversions below
  let mut output_exp:Vec<bool> = (*&vector_input[1..(1 + expwidth_input as usize)]).to_vec();

  
  if expwidth_input < expwidth_output {
    // upconverting exponent width

    for _i in 0..( expwidth_output - expwidth_input ) as usize {

      // println!("_i = {}", _i);

      // initialize increment vector
      let mut increment_vector:Vec<bool> = vec![false; output_exp.len() - 1];

      // add activation to top register of increment vector
      increment_vector.insert(0, true);

      // now add additional register at top with null to both
      output_exp.insert(0, false);
      increment_vector.insert(0, false);

      // now perform the increment
      output_exp = integer_add(&output_exp, &increment_vector);

    }

  } else if expwidth_input > expwidth_output {
    // the workflow is different
    // based on configuration of top two registers

    // 10 || 01 => remove top register, flip next register
    // 00 => return underflow
    // 11 => return overflow
    
    for _i in 0..( expwidth_input - expwidth_output ) as usize {
      
      if output_exp[0] && output_exp[1] {
        // return overflow
        // (all activations in exp and mantissa)
        
        output_mantissa = vec![true; output_mantissa_len as usize];

        output_exp = vec![true; expwidth_output as usize];

        // exit for loop
        break
        
      } else if !output_exp[0] && !output_exp[1] {
        // return underflow
        // (all null in exp and mantissa)
        output_mantissa = vec![false; output_mantissa_len as usize];

        output_exp = vec![false; expwidth_output as usize];

        // exit for loop
        break

      } else {
        // remove top register, flip second register
        // continue through for loop

        output_exp.remove(0);
        output_exp[0] = !output_exp[0];
        
      }
    }
  }

  // combine sets to the output_vector
  let output_vector:Vec<bool> = vec![output_sign, output_exp, output_mantissa].concat();
  
  return output_vector;
}


fn fp_convert_mantissa_precision(vector_input:&Vec<bool>, expwidth:u8, output_bitwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // *** pending integration of support functions
  // *** validation pending
  // *** support for diferent rounding conventions pending
  // (currently rounds to nearest)

  // converts vector_input to output vector
  // by increasing or decreasing mantissa register count
  // such that output_vector retains same expwidth basis
  // and output_vector has output_bitwidth basis

  // println!("fp_convert_mantissa_precision checkpoint 1");
  // println!("");

  // ****
  // we'll have convention that if user passes bitwidth_two = 0
  // then bitwidth_two retains input width 
  // which in context of fp_match_precision
  // is mantissa width realized after after fp_convert_exp_precision
  let mut final_output_bitwidth:u8 = output_bitwidth;
  if final_output_bitwidth == 0 {
    final_output_bitwidth = vector_input.len() as u8;
  }

  let mut output_vector:Vec<bool> = vector_input.clone();

  let input_bitwidth:u8 = vector_input.len() as u8;

  let input_mantissa_width:usize = vector_input.len() - 1 - expwidth as usize;

  // println!("fp_convert_mantissa_precision checkpoint 2");
  // println!("");

  if final_output_bitwidth > input_bitwidth {

    // println!("fp_convert_mantissa_precision checkpoint 3");
    // println!("");

    // in an alternate configuration
    // when increasing mantissa width
    // the populated bools could be randomly sampled
    let mut stochastic_rounding_sample:bool = false;

    if special_encodings.stochastic_rounding {
      stochastic_rounding_sample = sample_bool(vec![]);
    }

    for i in 0..(final_output_bitwidth - input_bitwidth) {
      output_vector.push(stochastic_rounding_sample);

    }

  } else if final_output_bitwidth < input_bitwidth {
  
    // for stochastic rounding consideraitons
    // this will be the vector used to weight the sampling
    let mantissa_rounding_bits:Vec<bool> = output_vector[final_output_bitwidth as usize ..].to_vec();
  
    // for stochastic rounding considerations
    // this output_vector has the "round down" configuration
    // if stochastic rounding applied we may subsequently round up by incrementing the mantissa
    // otherise rounding up is based on config of first bit in the mantissa_rounding_bits vector
    output_vector = output_vector[0..final_output_bitwidth as usize].to_vec();
  
    // stochastic_rounding_sample = false means retain the round down configuration
    // stochastic_rounding_sample = true will increment mantissa for the round up configuraiton
    let mut stochastic_rounding_sample:bool;
    if special_encodings.stochastic_rounding {
      stochastic_rounding_sample = sample_bool(mantissa_rounding_bits);
    } else {
      stochastic_rounding_sample = mantissa_rounding_bits[0];
    }
  
    // here we round up if applicable
    if stochastic_rounding_sample {
      output_vector = fp_increment_mantissa(&output_vector, expwidth);
    }
  }
  
  return output_vector;
}

fn fp_increment_mantissa(vector_input:&Vec<bool>, expwidth:u8) -> Vec<bool> {
  // *** pending validation
  
  // increments the final mantissa register
  // using an integer_add operation
  // unless mantissa is fully activated
  // in which case increments the exponent and deactivates mantissa
  // (and where if exp already saturated an increments returns mantissa saturation)

  // note that this operation is conducted without inspecting sign
  // so that e.g. 1.5 >> 2. and -1.5 >> -2.
  // I think that is correct way

  let output_sign:Vec<bool> = vec![vector_input[0]];

  let mut output_exp:Vec<bool> = (&vector_input[1..(expwidth as usize)]).to_vec();

  let mut output_mantissa:Vec<bool> = (&vector_input[(expwidth as usize)..]).to_vec();

  // check for the fully activate mantissa case
  if !output_mantissa.contains(&false) {
    
    let output_mantissa_len = output_mantissa.len();

    // reset output mantissa to false
    output_mantissa = vec![false; output_mantissa_len];

    // increment the exponent
    // first make sure exponent isn't already saturated
    if output_exp.contains(&false) {

      output_exp = integer_increment(&output_exp);
      
    } else {
      // otherwise will just return saturation value
      output_mantissa = vec![true; output_mantissa_len];
      
    }
    
  } else {
    // otherwise just increment the mantissa 
    output_mantissa = integer_increment(&output_mantissa);
    
  }

  // combine sets to the output_vector
  let output_vector:Vec<bool> = vec![output_sign, output_exp, output_mantissa].concat();
  
  return output_vector;
}


fn fp_compare(vector_one:&Vec<bool>, vector_two:&Vec<bool>, tolerance:&Vec<bool>, expwidth_one:u8, expwidth_two:u8, expwidth_tol:u8, special_encodings:&SpecialEncodings) -> bool {
  // compares two encoded floating point values 
  // potentiially of different precisions with known expwidths
  // to determine if within a tolerance
  // tolerance specified as encoded vector as well
  
  // note that the support function fp_match_precision
  // converts and returns a vector_one to basis of vector_two
  // so in cases where the expwidths do not match
  // this operation will translate to whichever basis 
  // has a larger expwidth for comaprison
  
  // note that special_encodings assume a particular basis
  // just as a starting point
  // for now we will only consider special_encodings
  // when expwidth_one == expwidth_two
  // *****and for mixed precision, 
  // will assume zero for the all null cases 
  // (need to put some more thought into how to do special encodings with mixed precision, have some thoughts just need to sketch them out, have placeholder for check_for_reserved_encoding_mixedprecision())
  
  // special encoding support pending
  // will assume zero for the all null cases 

  // uses criteria
  // abs(u - v)/abs(u) <= epsilon || 
  // abs(u - v)/abs(v) <= epsilon

  // when one of values is zero, uses criteria
  // abs(u - v) <= epsilon

  // hat tip documentation at boost.org
  // https://www.boost.org/doc/libs/1_61_0/libs/test/doc/html/boost_test/testing_tools/extended_comparison/floating_point/floating_points_comparison_theory.html

  // so workflow will be:
  // check if either input vector is zero
  // identify comparison basis
  // convert vector_one and vector_two to common basis
  // convert tolerance to common basis
  // apply comparison using formulas noted above

  let mut one_of_targets_is_zero:bool = false;

  // a future extension may check special encodings here
  // for now relying on default convention of zero is all false
  if !vector_one.contains(&true) || !vector_two.contains(&true) {
    one_of_targets_is_zero = true;
  }
  
  // basis vector is referring to basis
  // for conversion if mixed precision between inputs
  let mut basis_vector_is_one:bool = true;
  
  if expwidth_two > expwidth_one || (expwidth_two ==  expwidth_one && vector_two.len() > vector_one.len() ) {
    
    basis_vector_is_one = false;
    
  }

  let mut converted_vector_one:Vec<bool> = vec![];
  let mut converted_vector_two:Vec<bool> = vec![];

  let mut converted_expwidth:u8 = expwidth_one;
  let mut converted_bitwidth:u8 = vector_one.len() as u8;

  if basis_vector_is_one {

    converted_vector_two = 
      fp_match_precision(
        &vector_two, 
        expwidth_two, 
        vector_one.len() as u8, 
        expwidth_one, 
        &special_encodings
      );

    converted_vector_one = vector_one.clone();

    converted_expwidth = expwidth_one;
    converted_bitwidth = vector_one.len() as u8;

  } else {

    converted_vector_one = 
      fp_match_precision(
        &vector_one, 
        expwidth_one, 
        vector_two.len() as u8, 
        expwidth_two, 
        &special_encodings
      );

    converted_vector_two = vector_two.clone();

    converted_expwidth = expwidth_two;
    converted_bitwidth = vector_two.len() as u8;
  
  }

  // now convert tolerance to same basis
  let mut converted_tolerance:Vec<bool> = vec![];

  if expwidth_tol != converted_expwidth || converted_tolerance.len() != converted_bitwidth as usize {

    converted_tolerance = 
      fp_match_precision(
        &tolerance, 
        expwidth_tol, 
        converted_bitwidth, 
        converted_expwidth, 
        &special_encodings
      );
    
  } else {

    converted_tolerance = tolerance.clone();
    
  }

  // if one of inputs was zero result stored in one_of_targets_is_zero

  let mut returned_result = false;


  if !one_of_targets_is_zero {
    // when neither of targets is zero uses criteria
    // abs(u - v)/abs(u) <= epsilon || 
    // abs(u - v)/abs(v) <= epsilon

    let abs_one_minus_two = 
      fp_abs(
        &fp_subtract(
          &converted_vector_one, 
          &converted_vector_two, 
          converted_expwidth, 
          &special_encodings
        ), 
        &special_encodings
      );

    println!("ok here is before the fp_divide");
    println!("abs_one_minus_two.len() = {}", abs_one_minus_two.len());
    println!("converted_vector_one.len() = {}", converted_vector_one.len());
    println!("converted_expwidth = {}", converted_expwidth);
    println!("original expwidth_one = {}", expwidth_one);
    println!("original expwidth_two = {}", expwidth_two);

    // so I think what is needed is to perform a match rpecision in the denominator
    // fp_match_precision(&converted_vector_one, converted_expwidth, abs_one_minus_two.len() as u8, expwidth_two, &special_encodings)


    // let mut abs_one_minus_two_dividedby_abs_one = fp_divide(&abs_one_minus_two, &fp_abs(&converted_vector_one, &special_encodings), converted_expwidth, &special_encodings);

    let mut abs_one_minus_two_dividedby_abs_one = 
      fp_divide(&abs_one_minus_two, 
        &fp_abs(
          &fp_match_precision(
            &converted_vector_one, 
            converted_expwidth, 
            abs_one_minus_two.len() as u8, 
            expwidth_one, 
            &special_encodings
          ), 
          &special_encodings
        ), 
        converted_expwidth, 
        &special_encodings
      );


    // let mut abs_one_minus_two_dividedby_abs_two = fp_divide(&abs_one_minus_two, &fp_abs(&converted_vector_two, &special_encodings), converted_expwidth, &special_encodings);

    let abs_one_minus_two_dividedby_abs_two = 
      fp_divide(&abs_one_minus_two, 
        &fp_abs(
          &fp_match_precision(
            &converted_vector_two, 
            converted_expwidth, 
            abs_one_minus_two.len() as u8, 
            expwidth_two, 
            &special_encodings
          ), 
          &special_encodings
        ), 
        converted_expwidth, 
        &special_encodings
      );


    returned_result = 
      fp_greaterthan(
        &converted_tolerance, 
        &abs_one_minus_two_dividedby_abs_one, 
        converted_expwidth, 
        &special_encodings
      ) || fp_greaterthan(
        &converted_tolerance, 
        &abs_one_minus_two_dividedby_abs_two, 
        converted_expwidth, 
        &special_encodings
      );

  } else {
    // when one of values is zero, uses criteria
    // abs(u - v) <= epsilon

    let abs_one_minus_two = 
      fp_abs( 
        &fp_subtract(
          &converted_vector_one, 
          &converted_vector_two, 
          converted_expwidth, 
          &special_encodings
        ), 
        &special_encodings
      );

    returned_result = 
      fp_greaterthan(
        &converted_tolerance, 
        &abs_one_minus_two, 
        converted_expwidth, 
        &special_encodings
      );

  }

  return returned_result;
}


fn fp_fma(vector_one:&Vec<bool>, vector_two:&Vec<bool>, vector_three:&Vec<bool>, expwidth_onetwo:u8, expwidth_three:u8, special_encodings:&SpecialEncodings) -> (Vec<bool>, u8) {
  // performs fused multiply add operation (FMA)
  // assumes vector_one and vector_two in common precision
  // based on expwidth_onetwo
  // vector_three optionally in alternate precision
  // based on expwidth_three
  // for now will use convention that returned vector
  // has same precision basis as expwidth_three
  // and any variable returned mantissa width
  // (rounding can be applied by wrapping this function 
  // with a fp_match_precision or fp_convert_mantissa_precision call)

  // note that the scale parameter is used to apply
  // a power of 2 scaling to the product of vector_one and vector_two
  // in other words, this operation derives
  // vector_three += vector_one * vector_two * 2^scale
  
  // and assumes vector_one and vector_two share a common expwidth basis
  // and are received in a common bitwidth
  // and returns an output with expwidth_three basis
  // and a variable mantissa size to retain infinite precision

  // println!("fp_fma checkpoint 1");
  // println!("");

  
  // FMA applies formula
  // (vector_one times vector_two) plus vector_three

  // multiply vectors one and two
  // assumes common format
  // note that fp_multiply returns variable mantissa width for inf precision
  // let mut vector_one_times_two:Vec<bool> = fp_multiply(&vector_one, &vector_two, expwidth_onetwo, &special_encodings);

  // expwidth_onetwo is input exp width of vectors one and two which we assume are matche
  // expwidth_one_times_two is output expwidth from the multiply operation
  // which may be expanded in cases of overflow by fp_multiply_variable_exp

  let mut vector_one_times_two:Vec<bool>;
  let mut expwidth_one_times_two:u8;

  (vector_one_times_two, expwidth_one_times_two) = fp_multiply_variable_exp(&vector_one, &vector_two, expwidth_onetwo, &special_encodings);



  // println!("fp_fma checkpoint 2");
  // println!("");

  // the convention stated above is that output expwidth
  // is based on expwidth_three
  // however since both operations have variable mantissa widths
  // we'll call fp_match_precision with bitwidth_two = 0
  // which is a special case to allow variable output mantissa width
  // to retain infinite precision from multiplicaiton 
  // or any exp conversion

  // println!("before fp_match_precision");
  // println!("vector_one_times_two = {:?}", vector_one_times_two);
  // println!("expwidth_onetwo = {}", expwidth_onetwo);
  // println!("0 = {}", 0);
  // println!("expwidth_three = {}", expwidth_three);
  // println!("vector_one_times_two = fp_match_precision(&vector_one_times_two, expwidth_onetwo, 0, expwidth_three, &special_encodings);");
  // println!("");

  // in other words, put simply, here we'll match exp basis
  // between vector_one_times_two and the output
  vector_one_times_two = fp_match_precision(&vector_one_times_two, expwidth_one_times_two, 0, expwidth_three, &special_encodings);


  // println!("fp_fma checkpoint 3");
  // println!("");

  // initialize output vector based on received vector_three
  let mut output_vector:Vec<bool> = vector_three.clone();

  // ________

  // a future extension may integrate following into a wrapper for fp_match_precision
  // which would look cleaner anyway

  // now just in case this results in different mantissa widths
  // we'll convert mantissa width as well to match
  // vector_one_times_two and vector_three
  let vector_one_times_two_len = vector_one_times_two.len();
  let vector_three_len = vector_three.len();

  if vector_one_times_two_len > vector_three_len {

    // println!("fp_fma checkpoint 4");
    // println!("");

    output_vector = fp_convert_mantissa_precision(&output_vector, expwidth_three, vector_one_times_two_len as u8, &special_encodings);
    
  } else if vector_one_times_two_len < vector_three_len {

    // println!("fp_fma checkpoint 5");
    // println!("");

    vector_one_times_two = fp_convert_mantissa_precision(&vector_one_times_two, expwidth_three, vector_three_len as u8, &special_encodings);
  
  }

  // _________

  // great now that configurations are matched can apply fp_add
  // output_vector = fp_add(&output_vector, &vector_one_times_two, expwidth_three, &special_encodings);

  // return output_vector;

  // an extension could integrate the fp_add_variable_exp function
  // which would negate risk of accumulator overflow
  // there is a small distinction in that would need
  // to return a tuple as (output_vector, output_expwidth)
  // instead of just returning the output_vector
  // however could get around that if you fix the number of mantissa bits

  // println!("fp_fma checkpoint 6");
  // println!("");
  // println!("fp_add_variable_exp(&output_vector, &vector_one_times_two, expwidth_three, &special_encodings)");
  // println!("output_vector = {:?}", output_vector);
  // println!("vector_one_times_two = {:?}", vector_one_times_two);
  // println!("");

  // println!("before add operation");
  // println!("get_value(&sampled_array, expwidth, &special_encodings);")

  // (output_vector, output_expwidth) = 
  return fp_add_variable_exp(&output_vector, &vector_one_times_two, expwidth_three, &special_encodings)

  // would also need to update the function defintion
  // -> (Vec<bool>, u8)
  // and update the function call in fp_dot
  
}

fn fp_set_scaler(input_set:&Vec<bool>, scalar_vector:&Vec<bool>, bitwidth:u8, expwidth:u8, special_encodings:&SpecialEncodings) -> Vec<bool> {
  // accepts a vector of encoded vectors input_set
  // containing fp encoded floats of bitwidth and expwidth
  // used to scale each entry in that set 
  // by a constant power of two multiplier "exp_scalar"
  // where scalar_vector is expected as a bool vector
  // with consistent fp encoding basis as the input_set
  // and the scaling is achieved by fp_multiply
  // to each fp encoding entry of input_set_one
  // and where cases of exp overflow / underflow returns consistent conventions to fp basis

  let input_set_len:usize = input_set.len();
  let input_set_entry_count = input_set_len / bitwidth as usize;

  // for a first pass am going to populate a new output vector
  // presumably there may be benefit to editting inplace a mutable input vector
  // I just am not uyet comfortable with implications surrounding memory management for different computing conventions
  // and populating a new output vector is used as first pass to establish other conentions of the operation

  let mut output_set:Vec<bool> = vec![];

  for i in 0..input_set_entry_count {

    let target_entry:Vec<bool> = input_set[i..(i + bitwidth as usize)].to_vec();

    output_set.extend(
      fp_multiply(&target_entry, scalar_vector, expwidth, &special_encodings)
    );
    
  }

  return output_set;
}

  

fn fp_dot(input_set_one:&Vec<bool>, input_set_two:&Vec<bool>, bitwidth_onetwo:u8, vector_three:&Vec<bool>, expwidth_onetwo:u8, expwidth_three:u8, special_encodings:&SpecialEncodings) -> (Vec<bool>, u8) {
  
  // for floating point dot product
  // that accepts a vector of encoded vectors input_set_one
  // for dot product with input_set_two

  // assumes consistent floating point format and precision between those two inputs (per expwidth_onetwo and bitwidth_onetwo)
  // where those sets are basically concatinated encoding vectors
  // such that for a set of x floating point value pairs
  // input_set_one.len() = x * bitwidth_onetwo = input_set_two.len()
  
  // the vector_three serves as an accumulator vector
  // returns the result of the dot product from the accumulator


  let input_set_one_len:usize = input_set_one.len();
  let output_set_one_len:usize = input_set_one_len - bitwidth_onetwo as usize;
  
  let mut output_expwidth_three:u8 = expwidth_three;

  let mut returned_vector_three:Vec<bool> = vec![];

  if input_set_one_len > bitwidth_onetwo as usize {

    (returned_vector_three, output_expwidth_three) = fp_fma(&input_set_one[output_set_one_len..input_set_one_len].to_vec(), &input_set_two[output_set_one_len..input_set_one_len].to_vec(), &vector_three, expwidth_onetwo, output_expwidth_three, &special_encodings);


    (returned_vector_three, output_expwidth_three) = fp_dot(&input_set_one[..output_set_one_len].to_vec(), &input_set_two[..output_set_one_len].to_vec(), bitwidth_onetwo, &returned_vector_three, expwidth_onetwo, output_expwidth_three, &special_encodings);

    return (returned_vector_three, output_expwidth_three);

  } else {

    (returned_vector_three, output_expwidth_three) = fp_fma(&input_set_one[output_set_one_len..input_set_one_len].to_vec(), &input_set_two[output_set_one_len..input_set_one_len].to_vec(), &vector_three, expwidth_onetwo, output_expwidth_three, &special_encodings);

    return (returned_vector_three, output_expwidth_three);
    
  }

}
