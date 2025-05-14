package YoseUe_SATL

import chisel3._
import chisel3.util._
import chisel3.experimental._
import spatial_templates.pe._
import scala.math._


//Tournament ranking implementation

class EarlyTerminatorPE(id: ElemId, n_attr: Int, n_classes: Int, n_depths: Int, info_bit: Int, tree_bit: Int, max_votation: Double, n_ins: Int = 1) 
    extends PE(id) with WithFWConnection {
    val io = IO(new Bundle{
        val samples_in = Vec(n_ins,Flipped(Decoupled(new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit))))
        val samples_back = Vec(n_ins,Decoupled(new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit)))
        val sample_out = Decoupled(new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit))
    })

    val queues = VecInit(Seq.tabulate(n_ins)(i => Queue(io.samples_in(i), 2)))
    val max_votation_fp = max_votation.F(20.W, 6.BP)

    val next_power = ceil(log(n_classes) / log(2)).toInt
    val nextPow2 = pow(2, next_power).toInt

    // Compute total scores
    val total_scores = Wire(Vec(nextPow2, FixedPoint(16.W, 6.BP)))
    for (i <- 0 until nextPow2) {
        if (i < n_classes){
            val sum = queues.map(_.bits.scores(i)).reduce(_ +& _)
            total_scores(i) := sum
        }
        else{
            total_scores(i) := 0.F(16.W, 6.BP)
        }
    }

    val intermediate_valids = Reg(Vec(2 * next_power - 2, Bool()))
    val intermediate_dests = Reg(Vec(2 * next_power - 2, Bool()))
    val intermediate_overall_scores = Reg(Vec(2 * next_power - 2, Vec(n_classes, FixedPoint(16.W, 6.BP))))
    val intermediate_samples = Reg(Vec(2*next_power-1,new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit))) 
    val intermediate_samples_scores = Reg(Vec(2*next_power-1,Vec(n_ins, Vec(n_classes,FixedPoint(16.W,6.BP))))) 
    val candidates = Reg(Vec(2*next_power-2, Vec(nextPow2/2, FixedPoint(16.W, 6.BP))))
    val candidates_indexes = Reg(Vec(next_power-1, Vec(nextPow2/2, UInt(8.W))))
    val max1 = Reg(Vec(next_power-1,FixedPoint(16.W, 6.BP)))
    val max2 = Wire(FixedPoint(16.W, 6.BP))

    for (i <- 0 until 2*next_power-1){
        if (i==0){
            intermediate_valids(i) := queues(0).valid
            intermediate_overall_scores(i) := total_scores.take(n_classes)
            for(j <- 0 until n_ins){
                intermediate_samples_scores(i)(j) := queues(j).bits.scores
            }
            intermediate_dests(i) := queues.map(_.bits.dest).reduce(_ & _)
            intermediate_samples(i) := queues(0).bits
            for(j <- 0 until (nextPow2/2).toInt){
                candidates(0)(j) := Mux(total_scores(2*j) > total_scores(2*j + 1), total_scores(2*j), total_scores(2*j + 1))
                candidates_indexes(0)(j) := Mux(total_scores(2*j) > total_scores(2*j + 1), (2*j).U, (2*j + 1).U)
            }
        }else if (i<next_power-1){
            intermediate_valids(i) := intermediate_valids(i-1)
            intermediate_overall_scores(i) := intermediate_overall_scores(i-1)
            for(j <- 0 until n_ins){
                intermediate_samples_scores(i)(j) := intermediate_samples_scores(i-1)(j)
            }
            intermediate_dests(i) := intermediate_dests(i-1)
            intermediate_samples(i) := intermediate_samples(i-1)
            for(j <- 0 until nextPow2/((2*pow(2,i).toInt))){
                candidates(i)(j) := Mux(candidates(i-1)(2*j) > candidates(i-1)(2*j + 1), candidates(i-1)(2*j), candidates(i-1)(2*j + 1))
                candidates_indexes(i)(j) := Mux(candidates(i-1)(2*j) > candidates(i-1)(2*j + 1), candidates_indexes(i-1)(2*j), candidates_indexes(i-1)(2*j + 1))
            }
        }else if(i==next_power-1){
            intermediate_valids(i) := intermediate_valids(i-1)
            for(j <- 0 until n_ins){
                intermediate_samples_scores(i)(j) := intermediate_samples_scores(i-1)(j)
            }
            intermediate_dests(i) := intermediate_dests(i-1)
            intermediate_samples(i) := intermediate_samples(i-1)
            intermediate_overall_scores(i) := intermediate_overall_scores(i-1)
            max1(0) := Mux(candidates(i-1)(0) > candidates(i-1)(1), candidates(i-1)(0), candidates(i-1)(1))
            val loser = Mux(candidates(i-1)(0) > candidates(i-1)(1), candidates(i-1)(1), candidates(i-1)(0))
            val index = Mux(candidates(i-1)(0) > candidates(i-1)(1), candidates_indexes(i-1)(0), candidates_indexes(i-1)(1))
            when(index >= (nextPow2/2).U){
                for (j <- 0 until nextPow2/2){
                    if(j+nextPow2/2>=n_classes){
                        candidates(i)(j) := Mux(index === (j+nextPow2/2).U, loser, 0.F(16.W,6.BP))
                    }else{
                        candidates(i)(j) := Mux(index === (j+nextPow2/2).U, loser, intermediate_overall_scores(i-1)(j+nextPow2/2))
                    }
                }
            }.otherwise{
                for (j <- 0 until nextPow2/2){
                    candidates(i)(j) := Mux(index === j.U, loser, intermediate_overall_scores(i-1)(j)) 
                }
            }
        }else if(i<2*next_power-2){
            intermediate_valids(i) := intermediate_valids(i-1)
            for(j <- 0 until n_ins){
                intermediate_samples_scores(i)(j) := intermediate_samples_scores(i-1)(j)
            }
            intermediate_dests(i) := intermediate_dests(i-1)
            intermediate_samples(i) := intermediate_samples(i-1)
            intermediate_overall_scores(i) := intermediate_overall_scores(i-1)
            max1(i-next_power+1) := max1(i-next_power)
            for(j <- 0 until nextPow2/((2*pow(2,i-next_power+1)).toInt)){
                candidates(i)(j) := Mux(candidates(i-1)(2*j) > candidates(i-1)(2*j + 1), candidates(i-1)(2*j), candidates(i-1)(2*j + 1))
            }
        }else{
            for(j <- 0 until nextPow2/((2*pow(2,i-next_power+1)).toInt)){
                max2 := Mux(candidates(i-1)(2*j) > candidates(i-1)(2*j + 1), candidates(i-1)(2*j), candidates(i-1)(2*j + 1))
            }
        }
    }

    val reamining_votes = max_votation_fp - intermediate_overall_scores(2*next_power-3).map(x => x).reduce(_ + _)

    val distance = max1(next_power-2) - max2

    val condition = distance >= reamining_votes
    val dest = intermediate_dests(2*next_power-3)

    when(condition | dest){
        io.sample_out.valid := intermediate_valids(2*next_power-3)
        io.sample_out.bits.features := intermediate_samples(2*next_power-3).features
        io.sample_out.bits.weights := intermediate_samples(2*next_power-3).weights
        io.sample_out.bits.tree_to_exec := intermediate_samples(2*next_power-3).tree_to_exec
        io.sample_out.bits.shift := intermediate_samples(2*next_power-3).shift
        io.sample_out.bits.offset := intermediate_samples(2*next_power-3).offset
        io.sample_out.bits.dest := intermediate_samples(2*next_power-3).dest
        io.sample_out.bits.search_for_root := intermediate_samples(2*next_power-3).search_for_root
        io.sample_out.bits.last := intermediate_samples(2*next_power-3).last
        io.sample_out.bits.clock_cycles := intermediate_samples(2*next_power-3).clock_cycles
        for(i <- 0 until n_classes){
            io.sample_out.bits.scores(i) := intermediate_overall_scores(2*next_power-3)(i)
        }
        
        io.samples_back.map(_.valid := false.B)
        io.samples_back.map(_.bits := DontCare)

    }.otherwise{
        io.sample_out.valid := false.B
        io.sample_out.bits := DontCare
        
        for(i <- 0 until n_ins){
            io.samples_back(i).valid := intermediate_valids(2*next_power-3)
            io.samples_back(i).bits.features := intermediate_samples(2*next_power-3).features
            io.samples_back(i).bits.weights := intermediate_samples(2*next_power-3).weights
            io.samples_back(i).bits.tree_to_exec := intermediate_samples(2*next_power-3).tree_to_exec + 1.U
            io.samples_back(i).bits.shift := intermediate_samples(2*next_power-3).shift
            io.samples_back(i).bits.offset := intermediate_samples(2*next_power-3).tree_to_exec + 1.U
            io.samples_back(i).bits.dest := false.B
            io.samples_back(i).bits.search_for_root := intermediate_samples(2*next_power-3).search_for_root
            io.samples_back(i).bits.last := intermediate_samples(2*next_power-3).last
            io.samples_back(i).bits.clock_cycles := intermediate_samples(2*next_power-3).clock_cycles
            for(j <- 0 until n_classes){
                io.samples_back(i).bits.scores(j) := intermediate_samples_scores(2*next_power-3)(i)(j)
            }
        }
    }

    // Assuming paths with equal length, samples always arrive at the same clock cycle from each path
    queues.map(_.ready := io.sample_out.ready) 

    def linkToDest(backward_converter: BackwardConverter) {
        backward_converter.io.sample_in <> io.sample_out
    }

    def linkToDest(fi: FirstInterconnectPE, i:Int){ 
        io.samples_back(i) <> fi.io.sample_looping
    }
}

/*
//Linear Implementation with comparison one to one
class EarlyTerminatorPE(id: ElemId, n_attr: Int, n_classes: Int, n_depths: Int, info_bit: Int, tree_bit: Int, max_votation: Double, n_ins: Int = 1) 
    extends PE(id) with WithFWConnection {
    val io = IO(new Bundle{
        val samples_in = Vec(n_ins,Flipped(Decoupled(new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit))))
        val samples_back = Vec(n_ins,Decoupled(new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit)))
        val sample_out = Decoupled(new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit))
    })

    val queues = VecInit(Seq.tabulate(n_ins)(i => Queue(io.samples_in(i), 2)))
    val max_votation_fp = max_votation.F(18.W, 6.BP)

    // Compute total scores
    val total_scores = Wire(Vec(n_classes, FixedPoint(16.W, 6.BP)))
    for (i <- 0 until n_classes) {
        val sum = queues.map(_.bits.scores(i)).reduce(_ +& _)
        total_scores(i) := sum
    }
    val intermediate_valids = Reg(Vec((n_classes - 2), Bool()))
    val intermediate_dests = Reg(Vec((n_classes - 2), Bool()))
    val intermediate_samples = Reg(Vec(n_classes-2,new Sample(n_attr,n_classes,n_depths,info_bit,tree_bit))) 
    val intermediate_samples_scores = Reg(Vec(n_classes-2,Vec(n_ins, Vec(n_classes,FixedPoint(16.W,6.BP))))) 
    val intermediate_overall_scores = Reg(Vec(n_classes - 2, Vec(n_classes, FixedPoint(16.W, 6.BP))))
    val max1 = Reg(Vec(n_classes-2, FixedPoint(16.W, 6.BP)))
    val max2 = Reg(Vec(n_classes-2, FixedPoint(16.W, 6.BP)))
    val final_max1 = Wire(FixedPoint(16.W, 6.BP))
    val final_max2 = Wire(FixedPoint(16.W, 6.BP)) 

    for (i <- 0 until n_classes-1){
        if (i==0){
            intermediate_valids(i) := queues(0).valid
            intermediate_overall_scores(i) := total_scores
            for(j <- 0 until n_ins){
                intermediate_samples_scores(i)(j) := queues(j).bits.scores
            }
            intermediate_dests(i) := queues.map(_.bits.dest).reduce(_ & _)
            intermediate_samples(i) := queues(0).bits
            when(total_scores(0) > total_scores(1)){
                max1(0) := total_scores(0)
                max2(0) := total_scores(1)
            }.otherwise {
                max1(0) := total_scores(1)
                max2(0) := total_scores(0)
            }
        }else{
            if (i!=n_classes-2){
                intermediate_valids(i) := intermediate_valids(i-1)
                intermediate_overall_scores(i) := intermediate_overall_scores(i-1)
                for(j <- 0 until n_ins){
                    intermediate_samples_scores(i)(j) := intermediate_samples_scores(i-1)(j)
                }
                intermediate_dests(i) := intermediate_dests(i-1)
                intermediate_samples(i) := intermediate_samples(i-1) 
            }
            when(intermediate_overall_scores(i-1)(i+1) > max1(i-1)){
                final_max2 := max1(i-1)
                final_max1 := intermediate_overall_scores(i-1)(i+1)
            }.elsewhen(intermediate_overall_scores(i-1)(i+1) > max2(i-1)){
                final_max2 := intermediate_overall_scores(i-1)(i+1)
                final_max1 := max1(i-1)
            }.otherwise{
                final_max2 := max2(i-1)
                final_max1 := max1(i-1)
            }
        }
    }

    val reamining_votes = max_votation_fp - intermediate_overall_scores(n_classes-3).map(x => x).reduce(_ + _)

    val distance = final_max1 - final_max2

    val condition = distance >= reamining_votes
    val dest = intermediate_dests(n_classes-3)

    when(condition | dest){
        io.sample_out.valid := intermediate_valids(n_classes-3)
        io.sample_out.bits.features := intermediate_samples(n_classes-3).features
        io.sample_out.bits.weights := intermediate_samples(n_classes-3).weights
        io.sample_out.bits.tree_to_exec := intermediate_samples(n_classes-3).tree_to_exec
        io.sample_out.bits.shift := intermediate_samples(n_classes-3).shift
        io.sample_out.bits.offset := intermediate_samples(n_classes-3).offset
        io.sample_out.bits.dest := intermediate_samples(n_classes-3).dest
        io.sample_out.bits.search_for_root := intermediate_samples(n_classes-3).search_for_root
        io.sample_out.bits.last := intermediate_samples(n_classes-3).last
        io.sample_out.bits.clock_cycles := intermediate_samples(n_classes-3).clock_cycles
        for(i <- 0 until n_classes){
            io.sample_out.bits.scores(i) := intermediate_overall_scores(n_classes-3)(i)
        }
        
        io.samples_back.map(_.valid := false.B)
        io.samples_back.map(_.bits := DontCare)

    }.otherwise{
        io.sample_out.valid := false.B
        io.sample_out.bits := DontCare
        
        for(i <- 0 until n_ins){
            io.samples_back(i).valid := intermediate_valids(n_classes-3)
            io.samples_back(i).bits.features := intermediate_samples(n_classes-3).features
            io.samples_back(i).bits.weights := intermediate_samples(n_classes-3).weights
            io.samples_back(i).bits.tree_to_exec := intermediate_samples(n_classes-3).tree_to_exec + 1.U
            io.samples_back(i).bits.shift := intermediate_samples(n_classes-3).shift
            io.samples_back(i).bits.offset := intermediate_samples(n_classes-3).tree_to_exec + 1.U
            io.samples_back(i).bits.dest := false.B
            io.samples_back(i).bits.search_for_root := intermediate_samples(n_classes-3).search_for_root
            io.samples_back(i).bits.last := intermediate_samples(n_classes-3).last
            io.samples_back(i).bits.clock_cycles := intermediate_samples(n_classes-3).clock_cycles
            for(j <- 0 until n_classes){
                io.samples_back(i).bits.scores(j) := intermediate_samples_scores(n_classes-3)(i)(j)
            }
        }
    }

    // Assuming paths with equal length, samples always arrive at the same clock cycle from each path
    queues.map(_.ready := io.sample_out.ready) 

    def linkToDest(backward_converter: BackwardConverter) {
        backward_converter.io.sample_in <> io.sample_out
    }

    def linkToDest(fi: FirstInterconnectPE, i:Int){ 
        io.samples_back(i) <> fi.io.sample_looping
    }
}
*/
/*
//Old implementation. All in logic, one stage

val next_power = ceil(log(n_classes) / log(2)).toInt
val nextPow2 = pow(2, next_power).toInt

// Compute total scores
val total_scores = Wire(Vec(nextPow2, FixedPoint(16.W, 6.BP)))
for (i <- 0 until nextPow2) {
    if (i < n_classes){
        val sum = queues.map(_.bits.scores(i)).reduce(_ +& _)
        total_scores(i) := sum
    }
    else{
        total_scores(i) := 0.F(16.W, 6.BP)
    }
    //printf(p"SUM ${total_scores(i).asSInt}\n")
}

val max1 = Wire(FixedPoint(16.W,6.BP))
val max2 = Wire(FixedPoint(16.W,6.BP))

val half_cycle = ceil((2*next_power-1)/2.0).toInt-1
val loser = Wire(FixedPoint(16.W,6.BP))
val best_address = Wire(UInt(next_power.W))
var inputs = Wire(Vec(nextPow2, FixedPoint(16.W,6.BP)))
var inputs_indexes = Wire(Vec(nextPow2, UInt(next_power.W)))
for (j <- 0 until nextPow2){
    inputs(j) := total_scores(j)
    inputs_indexes(j) := j.U
}

for (i <- 0 until 2*next_power-1){

    if (i < half_cycle){
        val n_results = (nextPow2/pow(2,i+1)).toInt
        val results = Wire(Vec(n_results,FixedPoint(16.W,6.BP)))
        val indexes = Wire(Vec(n_results,UInt(next_power.W)))
        for (j <- 0 until n_results){
            results(j) := Mux(inputs(2*j)>inputs(2*j+1),inputs(2*j),inputs(2*j+1))
            indexes(j) := Mux(inputs(2*j)>inputs(2*j+1),inputs_indexes(2*j),inputs_indexes(2*j+1))
        }
        inputs = results
        inputs_indexes = indexes
    }else{
        if(i == half_cycle){
            max1 := Mux(inputs(0)>inputs(1),inputs(0),inputs(1))
            loser := Mux(inputs(0)<inputs(1),inputs(0),inputs(1))
            best_address := Mux(inputs(0)>inputs(1),inputs_indexes(0),inputs_indexes(1))
            val n_results = (nextPow2/2).toInt
            val results = Wire(Vec(n_results, FixedPoint(16.W,6.BP)))
            val boolean_check = Wire(Bool())
            boolean_check := Mux(best_address<((nextPow2/2).toInt).U,true.B,false.B)
            for (j <- 0 until n_results){
                results(j) := Mux(boolean_check,Mux(best_address===j.U,loser,total_scores(j)),Mux((best_address===(j + (nextPow2/2).toInt).U),loser,total_scores(j+(nextPow2/2).toInt)))
            }
            inputs = results
        }else{
            if (i<2*next_power-2){
                val n_results = (nextPow2/pow(2,i-half_cycle+1)).toInt
                val results = Wire(Vec(n_results,FixedPoint(16.W,6.BP)))
                for (j <- 0 until n_results){
                    results(j) := Mux(inputs(2*j)>inputs(2*j+1),inputs(2*j),inputs(2*j+1))
                }
                inputs = results
            }else{
                max2 := Mux(inputs(0)>inputs(1),inputs(0),inputs(1))
            }
        }
    }
    
}
*/

/*
//Old implementation all in logic without registers

val next_power = ceil(log(n_classes) / log(2)).toInt
val nextPow2 = pow(2, next_power).toInt

// Compute total scores
val total_scores = Wire(Vec(nextPow2, FixedPoint(16.W, 6.BP)))
for (i <- 0 until nextPow2) {
    if (i < n_classes){
        val sum = queues.map(_.bits.scores(i)).reduce(_ +& _)
        total_scores(i) := sum
    }
    else{
        total_scores(i) := 0.F(16.W, 6.BP)
    }
    //printf(p"SUM ${total_scores(i).asSInt}\n")
}
val max1 = Wire(FixedPoint(16.W,6.BP))
val max2 = Wire(FixedPoint(16.W,6.BP))

def parallelMax(vec: Vec[FixedPoint]): (FixedPoint,FixedPoint,UInt) = {
    var current = vec
    var indexes = Wire(Vec(nextPow2,UInt(next_power.W)))
    for (j <- 0 until nextPow2){
        indexes(j) := j.U
    }
    var loser = Wire(FixedPoint(16.W,6.BP))
    for (level <- 0 until log2Ceil(nextPow2)) {
        val next = Wire(Vec(current.length / 2, FixedPoint(16.W, 6.BP)))
        val next_indexes = Wire(Vec(current.length / 2, UInt(next_power.W)))
        if (level == log2Ceil(nextPow2)-1){
            next(0) := Mux(current(0) > current(1), current(0), current(1))
            loser := Mux(current(0) <= current(1), current(0), current(1))
            next_indexes(0) := Mux(current(0) > current(1), indexes(0),indexes(1))
        }else{
            for (i <- 0 until current.length / 2) {
                next(i) := Mux(current(2 * i) > current(2 * i + 1), current(2 * i), current(2 * i + 1))
                next_indexes(i) := Mux(current(2 * i) > current(2 * i + 1), indexes(2*i),indexes(2*i+1))
            }
        }
        current = next
        indexes = next_indexes
    }
    (current(0),loser,indexes(0))
}

def secondParallelMax(vec: Vec[FixedPoint]): FixedPoint = {
    var current = vec
    for (level <- 0 until log2Ceil(nextPow2/2)) {
        val next = Wire(Vec(current.length / 2, FixedPoint(16.W, 6.BP)))
        for (i <- 0 until current.length / 2) {
            next(i) := Mux(current(2 * i) > current(2 * i + 1), current(2 * i), current(2 * i + 1))
        }
        current = next
    }
    current(0)
}

val index = Wire(UInt(next_power.W))
val loser = Wire(FixedPoint(16.W,6.BP))
val temp = parallelMax(total_scores) // Compute the first max

max1 := temp._1
loser := temp._2
index := temp._3

// **Step 2: Find Second Max Using a Similar Reduction Tree**
val losers = Wire(Vec(nextPow2/2, FixedPoint(16.W, 6.BP)))

// Populate losers with the values of the half side of the winners, but replace max1 with 0
when(index>=(nextPow2/2).U){
    for (i <- 0 until nextPow2/2) {
        losers(i) := Mux(index === (i+nextPow2/2).U, loser, total_scores(i+nextPow2/2)) 
    }
}.otherwise{
    for (i <- 0 until nextPow2/2) {
        losers(i) := Mux(index === i.U, loser, total_scores(i)) 
    }
}
max2 := secondParallelMax(losers) // Compute second max on modified input

*/